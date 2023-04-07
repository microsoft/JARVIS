require 'nn'

require 'cunn'
require 'dpnn'

require 'fbnn'
require 'fbcunn'

require 'optim'

paths.dofile('torch-TripletEmbedding/TripletEmbedding.lua')

if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   modelAnchor = torch.load(opt.retrain)
else
   paths.dofile(opt.modelDef)
   modelAnchor = createModel(opt.nGPU)
end

modelPos = modelAnchor:clone('weight', 'bias', 'gradWeight', 'gradBias')
modelNeg = modelAnchor:clone('weight', 'bias', 'gradWeight', 'gradBias')

model = nn.ParallelTable()
model:add(modelAnchor)
model:add(modelPos)
model:add(modelNeg)

alpha = 0.2
criterion = nn.TripletEmbeddingCriterion(alpha)

model = model:cuda()
criterion:cuda()

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

collectgarbage()
