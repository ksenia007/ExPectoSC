
import torch
import torch.nn as nn

from functools import reduce
from torch.autograd import Variable

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


deepsea_beluga_2002_cpu = nn.Sequential( # Sequential,
	nn.Sequential( # Sequential,
		nn.Conv2d(4,320,(1, 8)),
		nn.ReLU(),
		nn.Conv2d(320,320,(1, 8)),
		nn.ReLU(),
		nn.Dropout(0.2),
		nn.MaxPool2d((1, 4),(1, 4)),
		nn.Conv2d(320,480,(1, 8)),
		nn.ReLU(),
		nn.Conv2d(480,480,(1, 8)),
		nn.ReLU(),
		nn.Dropout(0.2),
		nn.MaxPool2d((1, 4),(1, 4)),
		nn.Conv2d(480,640,(1, 8)),
		nn.ReLU(),
		nn.Conv2d(640,640,(1, 8)),
		nn.ReLU(),
	),
	nn.Sequential( # Sequential,
		nn.Dropout(0.5),
		Lambda(lambda x: x.view(x.size(0),-1)), # Reshape,
		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)), # Linear,
		nn.ReLU(),
		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,2002)), # Linear,
	),
	nn.Sigmoid(),
)