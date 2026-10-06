"""Read-only Hopper NVML GPM monitoring; no CUDA context or policy changes."""
import argparse, ctypes as C, json, math, signal, time
from pathlib import Path
METRICS={1:'graphics_active_pct',2:'sm_active_pct',3:'sm_occupancy_pct',4:'integer_active_pct',5:'tensor_active_pct',7:'hmma_active_pct',10:'dram_bandwidth_pct',12:'fp32_active_pct',13:'fp16_active_pct'}
class Info(C.Structure):
    _fields_=[('shortName',C.c_char_p),('longName',C.c_char_p),('unit',C.c_char_p)]
class Metric(C.Structure):
    _fields_=[('metricId',C.c_uint),('nvmlReturn',C.c_int),('value',C.c_double),('metricInfo',Info)]
class MetricsV1(C.Structure):
    # The installed NVML v1 header defines the fixed v1 array as 98 entries.
    _fields_=[('version',C.c_uint),('numMetrics',C.c_uint),('sample1',C.c_void_p),('sample2',C.c_void_p),('metrics',Metric*98)]
class Support(C.Structure):
    _fields_=[('version',C.c_uint),('isSupportedDevice',C.c_uint)]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--uuid',required=True)
    parser.add_argument('--interval',type=float,default=0.25)
    parser.add_argument('--count',type=int,default=0)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not math.isfinite(args.interval) or args.interval<=0.1:
        raise ValueError('NVML GPM sample interval must be finite and exceed 100 ms')
    if args.count<0:raise ValueError('sample count must be nonnegative')
    lib=C.CDLL('libnvidia-ml.so.1')
    signatures={
        'nvmlInit_v2': [], 'nvmlShutdown': [],
        'nvmlDeviceGetHandleByUUID': [C.c_char_p,C.POINTER(C.c_void_p)],
        'nvmlGpmSampleAlloc': [C.POINTER(C.c_void_p)],
        'nvmlGpmSampleFree': [C.c_void_p],
        'nvmlGpmSampleGet': [C.c_void_p,C.c_void_p],
        'nvmlGpmQueryDeviceSupport': [C.c_void_p,C.POINTER(Support)],
        'nvmlGpmMetricsGet': [C.POINTER(MetricsV1)],
        'nvmlGpmQueryIfStreamingEnabled': [C.c_void_p,C.POINTER(C.c_uint)],
    }
    for name,types in signatures.items():
        function=getattr(lib,name);function.argtypes=types;function.restype=C.c_int
    lib.nvmlErrorString.argtypes=[C.c_int]
    lib.nvmlErrorString.restype=C.c_char_p
    def check(code):
        if code:raise RuntimeError(f'NVML {code}: {lib.nvmlErrorString(code).decode()}')
    check(lib.nvmlInit_v2())
    device=C.c_void_p(); samples=[]
    stop=False
    def shutdown(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,shutdown)
    signal.signal(signal.SIGINT,shutdown)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    try:
        check(lib.nvmlDeviceGetHandleByUUID(args.uuid.encode(),C.byref(device)))
        support=Support(1,0); check(lib.nvmlGpmQueryDeviceSupport(device,C.byref(support)))
        if not support.isSupportedDevice:raise RuntimeError('NVML GPM unsupported')
        for _ in range(2):
            sample=C.c_void_p();check(lib.nvmlGpmSampleAlloc(C.byref(sample)));samples.append(sample)
        state=C.c_uint();check(lib.nvmlGpmQueryIfStreamingEnabled(device,C.byref(state)))
        with out.open('w') as file:
            file.write(json.dumps({'type':'metadata','uuid':args.uuid,'interval_seconds':args.interval,'streaming_state_before':state.value,'metric_definitions':METRICS,'struct_version':1,'struct_bytes':C.sizeof(MetricsV1),'scope':'Device-wide interval counters; attribution requires isolated workload. No GPM policy setters used.'})+'\n');file.flush()
            previous=None; index=0
            while not stop and (not args.count or index<args.count):
                current=samples[index%2]
                code=lib.nvmlGpmSampleGet(device,current)
                timestamp=time.time()
                record={'type':'sample','utc_epoch':timestamp,'sample_status':code}
                if code==0 and previous is not None:
                    request=MetricsV1();request.version=1;request.numMetrics=len(METRICS)
                    request.sample1=previous;request.sample2=current
                    for slot,metric_id in enumerate(METRICS):request.metrics[slot].metricId=metric_id
                    result=lib.nvmlGpmMetricsGet(C.byref(request));record['metrics_status']=result
                    if result==0:
                        record['metrics']={METRICS[m.metricId]:{'status':m.nvmlReturn,'raw_value':m.value if math.isfinite(m.value) else None,'value':m.value if m.nvmlReturn==0 and math.isfinite(m.value) and 0<=m.value<=100 else None} for m in request.metrics[:len(METRICS)]}
                file.write(json.dumps(record,allow_nan=False)+'\n');file.flush()
                previous=current if code==0 else None
                index+=1;time.sleep(args.interval)
            state_after=C.c_uint();check(lib.nvmlGpmQueryIfStreamingEnabled(device,C.byref(state_after)))
            file.write(json.dumps({'type':'end','streaming_state_after':state_after.value})+'\n')
    finally:
        for sample in samples:check(lib.nvmlGpmSampleFree(sample))
        check(lib.nvmlShutdown())
if __name__=='__main__':main()
