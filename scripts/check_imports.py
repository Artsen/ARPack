import importlib, traceback
paths = ["src.ingest","scripts.hydrate_and_chunk","src.chunk","src.embed","scripts.build_index"]
errs = False
for p in paths:
    try:
        importlib.import_module(p)
        print("OK:", p)
    except Exception as e:
        print("ERR importing", p, ":", e)
        traceback.print_exc()
        errs = True
if errs:
    raise SystemExit(1)
