import json, os
# Parameter files for the training-health alarms validation runs (plan V-2..V-5),
# derived from the LR-schedule A/B recipes; written next to this script.
V = os.path.dirname(os.path.abspath(__file__)) + '/'
E = os.path.normpath(V + '../20261005-lr-schedule-ab') + '/'
def write(src, name, extra):
    d=json.load(open(E+src))
    d.update(extra)
    json.dump(d, open(V+name,'w'), indent=2, sort_keys=True)
write('parameters-C.json','params-C.json',{})
write('parameters-C.json','params-C-stop.json',{'training_health_action_illegal_mass':1})
write('parameters-A.json','params-A.json',{})
write('parameters-A.json','params-A-off.json',{'training_health_alarms_enabled':False})
write('parameters-A.json','params-A-on50.json',{'training_health_check_interval_steps':50})
print('ok')
