import lldb
S="/private/tmp/claude-501/-Users-andrew-cursor-drews-chess-machine/b355da5a-0481-4110-8826-7e5977bb28e4/scratchpad"
def ev(frame, e):
    opt=lldb.SBExpressionOptions(); opt.SetLanguage(lldb.eLanguageTypeObjC); opt.SetTimeoutInMicroSeconds(5_000_000)
    v=frame.EvaluateExpression(e, opt)
    return v
def run(debugger, command, result, d):
    frame=debugger.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame()
    out=open(S+"/mtl_objects.txt","w")
    for a in [l.strip() for l in open(S+"/heapaddrs.txt")]:
        vals=[ev(frame,f"(unsigned long long)[(id){a} {k}]").GetValueAsUnsigned() for k in ("size","usedSize","currentAllocatedSize")]
        lab=ev(frame,f"(id)[(id){a} label]").GetObjectDescription()
        out.write(f"HEAP {a} size={vals[0]} used={vals[1]} cur={vals[2]} label={lab}\n")
    for a in [l.strip() for l in open(S+"/bufaddrs.txt")]+[l.strip() for l in open(S+"/pooladdrs.txt")]:
        n=ev(frame,f"(unsigned long long)[(id){a} length]").GetValueAsUnsigned()
        if n>=1048576:
            lab=ev(frame,f"(id)[(id){a} label]").GetObjectDescription()
            hp=ev(frame,f"(unsigned long long)[(id){a} heap]").GetValueAsUnsigned()
            off=ev(frame,f"(unsigned long long)[(id){a} heapOffset]").GetValueAsUnsigned()
            out.write(f"BUF {a} len={n} heap={hp:#x} off={off} label={lab}\n")
    out.close()
def __lldb_init_module(debugger, d):
    debugger.HandleCommand("command script add -f mtl_probe.run mtlprobe")
