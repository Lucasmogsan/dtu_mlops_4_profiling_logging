import torch
import torchvision.models as models
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity

model = models.resnet18()  # or any other net - eg models.resnet34(), models.resnet18() etc.
inputs = torch.randn(5, 3, 224, 224)


### Profile CPU usage:
# Profile a single iteration
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    model(inputs)
# Profile multiple iterations to remove dependency on what background processes that are running on your computer
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for i in range(10):
        model(inputs)
        prof.step()
        
# Print from the 'prof' object created above:
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
# Export to file

prof.export_chrome_trace("trace.json")
# Using tensorboard:
with profile(activities=[ProfilerActivity.CPU], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/resnet18")) as prof:
    for i in range(10):
        model(inputs)
        prof.step()







### Profile memory usage:
profile_memory = False
if profile_memory:
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        model(inputs)
    # print info
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))


### Profile GPU usage (if available):
if torch.cuda.is_available():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        model(inputs)
    # print info


