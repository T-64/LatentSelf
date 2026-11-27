"""GPU-host-only residual reader. Requests and token activations stay in memory."""
import argparse
from hashlib import sha256
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse,parse_qs

from .sae_io import Pool,chunks


class Reader:
    def __init__(self,model_path,sae_path,reader,sae_repo,precision,formula):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
        self.torch = torch
        self.reader,self.sae_repo,self.precision,self.formula = reader,sae_repo,precision,formula
        self.sae_path = Path(sae_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=True)
        options = dict(local_files_only=True,dtype=torch.bfloat16,device_map="auto")
        if precision=="bf16-offload":
            options["max_memory"] = {0:"36GiB","cpu":"100GiB"}
        elif precision=="int8":
            options["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,**options).eval()
        base = self.model.model
        self.layers = getattr(base,"language_model",base).layers
        self.device = self.model.get_input_embeddings().weight.device
        self.loaded = {}

    def weights(self,layer):
        if layer not in self.loaded:
            self.loaded.clear()  # one large SAE resides in GPU memory at a time
            state = self.torch.load(self.sae_path/f"layer{layer}.sae.pt",map_location="cpu",weights_only=True)
            self.loaded[layer] = {k:state[k].to(self.device,dtype=self.torch.bfloat16)
                                  for k in ("W_enc","W_dec","b_enc","b_dec")}
        return self.loaded[layer]

    def encode_layer(self,text,layer,*,formula=None,feature=None,experiment=False):
        torch = self.torch
        formula = formula or self.formula
        if formula not in ("card","centered") or not 0<=layer<len(self.layers):
            raise ValueError("invalid layer or formula")
        inputs = self.tokenizer(text,return_offsets_mapping=True,add_special_tokens=True)
        ids = inputs["input_ids"]
        spans = list(chunks(len(ids)))
        weights = self.weights(layer)
        if feature is not None and not 0<=feature<len(weights["b_enc"]):
            raise ValueError("unknown feature")
        pooled = Pool()
        tokens = [0.0]*len(ids) if feature is not None else None
        sum_x = None
        err = square = ce_change = 0.
        count = ce_count = active = 0
        covered = 1
        for begin,end in spans:
            skip = max(0,covered-begin)
            batch = torch.tensor([ids[begin:end]],device=self.device)
            captured = {}
            def hook(module,args,out):
                original = out[0] if isinstance(out,tuple) else out
                hidden = original[0].to(weights["b_enc"].device,dtype=weights["b_enc"].dtype)
                x = hidden if formula=="card" else hidden-weights["b_dec"]
                pre = x@weights["W_enc"].T+weights["b_enc"]
                if formula=="centered":
                    pre.relu_()
                values,indices = pre.topk(50,dim=-1)
                sparse = torch.zeros_like(pre).scatter_(1,indices,values)
                rebuilt = sparse@weights["W_dec"].T+weights["b_dec"]
                captured.update(hidden=hidden,rebuilt=rebuilt,values=values,indices=indices)
                if experiment:
                    replacement = rebuilt.to(original.device,dtype=original.dtype).unsqueeze(0)
                    replacement[:,0] = original[:,0]
                    return (replacement,*out[1:]) if isinstance(out,tuple) else replacement
            with torch.inference_mode(),ExitStack() as stack:
                reference = self.model(batch,use_cache=False).logits if experiment else None
                handle = self.layers[layer].register_forward_hook(hook)
                stack.callback(handle.remove)
                if experiment:
                    logits = self.model(batch,use_cache=False).logits
                    start = max(0,skip-1)
                    target = batch[:,start+1:].reshape(-1)
                    if target.numel():
                        def ce(logits):
                            return torch.nn.functional.cross_entropy(logits[:,start:-1].reshape(-1,logits.shape[-1]).float(),
                                                                     target.to(logits.device),reduction="sum")
                        ce_change += float(ce(logits)-ce(reference))
                        ce_count += target.numel()
                else:
                    self.model.model(input_ids=batch,use_cache=False)
                values,indices = captured["values"],captured["indices"]
                pooled.add(begin,indices.cpu().tolist(),values.float().cpu().tolist())
                if tokens is not None:
                    token_values = (values.masked_fill(indices!=feature,0).amax(1)).float().cpu().tolist()
                    for j,value in enumerate(token_values,begin):
                        if j:
                            tokens[j] = max(tokens[j],value)
                h = captured["hidden"][skip:].float()
                r = captured["rebuilt"][skip:].float()
                if len(h):
                    total = h.sum(dim=0)
                    sum_x = total if sum_x is None else sum_x+total
                    square += float(h.square().sum())
                    err += float((h-r).square().sum())
                    active += int((values[skip:]>0).sum())
                    count += len(h)
                covered = end
        denom = max(0,square-float(sum_x.square().sum())/count) if count else 0
        value = dict(reader=self.reader,sae_repo=self.sae_repo,precision=self.precision,formula=formula,
                     n_tokens=len(ids),n_chunks=len(spans),fvu=err/denom if denom else 0,
                     sse=err,variance=denom,delta_ce=ce_change/ce_count if ce_count else None,
                     ce_tokens=ce_count,l0=active/count if count else 0,active_features=sorted(pooled.best) if experiment else None,
                     features=pooled.strongest())
        if tokens is not None:
            value.update(token_acts=tokens,offsets=inputs["offset_mapping"])
        return value

    def neighbors(self,layer,feature):
        torch = self.torch
        w = self.weights(layer)["W_dec"]
        if not 0<=feature<w.shape[1]:
            raise ValueError("unknown feature")
        target = w[:,feature].float()
        scores = torch.cat([torch.nn.functional.cosine_similarity(part.float(),target[:,None],dim=0)
                            for part in w.split(1024,dim=1)])
        scores[feature] = -2
        v,ix = scores.topk(min(10,len(scores)-1))
        return list(zip(ix.tolist(),v.tolist()))


def serve(reader,port,baseline=None):
    public = {}
    if baseline:
        with Path(baseline).open(encoding="utf-8") as source:
            for line in source:
                obj = json.loads(line)
                public[obj["id"]] = obj
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def send_json(self,status,value):
            body = json.dumps(value,allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send_json(200 if self.path=="/health" else 404,{"reader":reader.reader})

        def do_POST(self):
            url = urlparse(self.path)
            if url.path not in ("/encode","/neighbors","/baseline"):
                return self.send_json(404,{"error":"unknown route"})
            try:
                length = int(self.headers["Content-Length"])
                if not 0<length<=32*1024*1024:
                    return self.send_json(413,{"error":"request exceeds 32 MiB"})
                payload = json.loads(self.rfile.read(length))
                if url.path=="/baseline":
                    if not public:
                        return self.send_json(409,{"error":"public baseline sample is unavailable"})
                    if payload.get("list"):
                        result = [dict(id=uid,source=item["source"],hash=sha256(item["text"].encode()).hexdigest())
                                  for uid,item in public.items()]
                    else:
                        result = reader.encode_layer(public[payload["id"]]["text"],payload["layer"])
                elif url.path=="/neighbors":
                    result = reader.neighbors(payload["layer"],payload["feature"])
                else:
                    feature = payload.get("feature")
                    if "per_token" in parse_qs(url.query) and feature is None:
                        raise ValueError("feature is required for highlights")
                    result = [{str(layer):reader.encode_layer(text,layer,feature=feature,formula=payload.get("formula"),
                               experiment=bool(payload.get("experiment"))) for layer in payload["layers"]}
                              for text in payload["texts"]]
                self.send_json(200,dict(results=result))
            except (ValueError,KeyError,IndexError,TypeError):
                self.send_json(400,{"error":"invalid reader request"})
            except RuntimeError:
                self.send_json(503,{"error":"reader could not finish computation"})
    HTTPServer(("127.0.0.1",port),Handler).serve_forever()


def main():
    p = argparse.ArgumentParser()
    for name in ("model-path","sae-path","reader","sae-repo"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--precision",choices=("bf16","bf16-offload","int8"),required=True)
    p.add_argument("--formula",choices=("card","centered"),default="card")
    p.add_argument("--port",type=int,default=8003)
    p.add_argument("--baseline")
    a = p.parse_args()
    serve(Reader(a.model_path,a.sae_path,a.reader,a.sae_repo,a.precision,a.formula),a.port,a.baseline)


if __name__=="__main__":
    main()
