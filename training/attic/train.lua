-- Copyright 2015-2016 Carnegie Mellon University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.


-- This code samples images and trains a triplet network with the
-- following steps, which are referenced inline.
--
-- [Step 1]
-- Sample at most opt.peoplePerBatch * opt.imagesPerPerson
-- images by choosing random people and images from the
-- training set.
--
-- [Step 2]
-- Compute the embeddings of all of these images by doing forward
-- passs with the current state of a network.
-- This is done offline and the network is not modified.
-- Since not all of the images will fit in GPU memory, this is
-- split into minibatches.
--
-- [Step 3]
-- Select the semi-hard triplets as described in the FaceNet paper.
--
-- [Step 4]
-- Google is able to do a single forward and backward pass to process
-- all the triplets and update the network's parameters at once since
-- they use a distributed system.
-- With a memory-limited GPU, OpenFace uses smaller mini-batches and
-- does many forward and backward passes to iteratively update the
-- network's parameters.
--
--
--
-- Some other useful references for models with shared weights are:
--
--  1. Weinberger, K. Q., & Saul, L. K. (2009).
--     Distance metric learning for large margin
--     nearest neighbor classification.
--     The Journal of Machine Learning Research, 10, 207-244.
--
--     http://machinelearning.wustl.edu/mlpapers/paper_files/jmlr10_weinberger09a.pdf
--
--
--     Citation from the FaceNet paper on their motivation for
--     using the triplet loss.
--
--
--  2. Chopra, S., Hadsell, R., & LeCun, Y. (2005, June).
--     Learning a similarity metric discriminatively, with application
--     to face verification.
--     In Computer Vision and Pattern Recognition, 2005. CVPR 2005.
--     IEEE Computer Society Conference on (Vol. 1, pp. 539-546). IEEE.
--
--     http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf
--
--
--     The idea is to just look at pairs of images at a time
--     rather than triplets, which they train with two networks
--     in parallel with shared weights.
--
--  3. Hoffer, E., & Ailon, N. (2014).
--     Deep metric learning using Triplet network.
--     arXiv preprint arXiv:1412.6622.
--
--     http://arxiv.org/abs/1412.6622
--
--
--     Not used in OpenFace or FaceNet, but another view of triplet
--     networks that provides slightly more details about training using
--     three networks with shared weights.
--     The code uses Torch and is available on GitHub at
--     https://github.com/eladhoffer/TripletNet

require 'optim'
require 'fbnn'
require 'image'

paths.dofile("OpenFaceOptim.lua")


local optimMethod = optim.adadelta
local optimState = {} -- Use for other algorithms like SGD
local optimator = OpenFaceOptim(model, optimState)

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

local batchNumber
local triplet_loss

function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()
   model:cuda() -- get it back on the right GPUs.

   local tm = torch.Timer()
   triplet_loss = 0

   local i = 1
   while batchNumber < opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         function()
            -- [Step 1]: Sample people/images from the dataset.
            local inputs, numPerClass = trainLoader:samplePeople(opt.peoplePerBatch,
                                                                 opt.imagesPerPerson)
            inputs = inputs:float()
            numPerClass = numPerClass:float()
            return sendTensor(inputs), sendTensor(numPerClass)
         end,
         trainBatch
      )
      if i % 5 == 0 then
         donkeys:synchronize()
      end
      i = i + 1
   end

   donkeys:synchronize()
   cutorch.synchronize()

   triplet_loss = triplet_loss / batchNumber

   trainLogger:add{
      ['avg triplet loss (train set)'] = triplet_loss,
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average triplet loss (per batch): %.2f',
                       epoch, tm:time().real, triplet_loss))
   print('\n')

   collectgarbage()

   local function sanitize(net)
      net:apply(function (val)
            for name,field in pairs(val) do
               if torch.type(field) == 'cdata' then val[name] = nil end
               if name == 'homeGradBuffers' then val[name] = nil end
               if name == 'input_gpu' then val['input_gpu'] = {} end
               if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
               if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
               if (name == 'output' or name == 'gradInput')
               and torch.type(field) == 'torch.CudaTensor' then
                  cutorch.withDevice(field:getDevice(), function() val[name] = field.new() end)
               end
            end
      end)
   end
   sanitize(model)
   torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'),
              model.modules[1]:float())
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
   collectgarbage()
end -- of train()

local inputsCPU = torch.FloatTensor()
local numPerClass = torch.FloatTensor()

local timer = torch.Timer()
function trainBatch(inputsThread, numPerClassThread)
   if batchNumber >= opt.epochSize then
      return
   end

   cutorch.synchronize()
   timer:reset()
   receiveTensor(inputsThread, inputsCPU)
   receiveTensor(numPerClassThread, numPerClass)

   -- [Step 2]: Compute embeddings.
   local numImages = inputsCPU:size(1)
   local embeddings = torch.Tensor(numImages, 128)
   local singleNet = model.modules[1]
   local beginIdx = 1
   local inputs = torch.CudaTensor()
   while beginIdx <= numImages do
      local endIdx = math.min(beginIdx+opt.batchSize-1, numImages)
      local range = {{beginIdx,endIdx}}
      local sz = inputsCPU[range]:size()
      inputs:resize(sz):copy(inputsCPU[range])
      local reps = singleNet:forward(inputs):float()
      embeddings[range] = reps

      beginIdx = endIdx + 1
   end
   assert(beginIdx - 1 == numImages)

   -- [Step 3]: Select semi-hard triplets.
   local numTrips = numImages - opt.peoplePerBatch
   local as = torch.Tensor(numTrips, inputs:size(2),
                           inputs:size(3), inputs:size(4))
   local ps = torch.Tensor(numTrips, inputs:size(2),
                           inputs:size(3), inputs:size(4))
   local ns = torch.Tensor(numTrips, inputs:size(2),
                           inputs:size(3), inputs:size(4))

   function dist(emb1, emb2)
      local d = emb1 - emb2
      return d:cmul(d):sum()
   end

   local tripIdx = 1
   local shuffle = torch.randperm(numTrips)
   local embStartIdx = 1
   local nRandomNegs = 0
   for i = 1,opt.peoplePerBatch do
      local n = numPerClass[i]
      for j = 1,n-1 do
         local aIdx = embStartIdx
         local pIdx = embStartIdx+j
         as[shuffle[tripIdx]] = inputsCPU[aIdx]
         ps[shuffle[tripIdx]] = inputsCPU[pIdx]

         -- Select a semi-hard negative that has a distance
         -- further away from the positive exemplar.
         local posDist = dist(embeddings[aIdx], embeddings[pIdx])

         local selNegIdx = embStartIdx
         while selNegIdx >= embStartIdx and selNegIdx <= embStartIdx+n-1 do
            selNegIdx = (torch.random() % numImages) + 1
         end
         local selNegDist = dist(embeddings[aIdx], embeddings[selNegIdx])
         local randomNeg = true
         for k = 1,numImages do
            if k < embStartIdx or k > embStartIdx+n-1 then
               local negDist = dist(embeddings[aIdx], embeddings[k])
               if posDist < negDist and negDist < selNegDist and
                     math.abs(posDist-negDist) < alpha then
                  randomNeg = false
                  selNegDist = negDist
                  selNegIdx = k
               end
            end
         end
         if randomNeg then
            nRandomNegs = nRandomNegs + 1
         end

         ns[shuffle[tripIdx]] = inputsCPU[selNegIdx]

         tripIdx = tripIdx + 1
      end
      embStartIdx = embStartIdx + n
   end
   assert(embStartIdx - 1 == numImages)
   assert(tripIdx - 1 == numTrips)
   print(('  + (nRandomNegs, nTrips) = (%d, %d)'):format(nRandomNegs, numTrips))


   -- [Step 4]: Upate network parameters.
   beginIdx = 1
   local asCuda = torch.CudaTensor()
   local psCuda = torch.CudaTensor()
   local nsCuda = torch.CudaTensor()

   -- Return early if the loss is 0 for `numZeros` iterations.
   local numZeros = 4
   local zeroCounts = torch.IntTensor(numZeros):zero()
   local zeroIdx = 1

   -- Return early if the loss shrinks too much.
   -- local firstLoss = nil

   -- TODO: Should be <=, but batches with just one image cause errors.
   while beginIdx < numTrips do
      local endIdx = math.min(beginIdx+opt.batchSize, numTrips)

      local range = {{beginIdx,endIdx}}
      local sz = as[range]:size()
      asCuda:resize(sz):copy(as[range])
      psCuda:resize(sz):copy(ps[range])
      nsCuda:resize(sz):copy(ns[range])
      local err, _ = optimator:optimizeTriplet(optimMethod,
                                               {asCuda, psCuda, nsCuda},
                                               criterion)

      cutorch.synchronize()
      batchNumber = batchNumber + 1
      print(('Epoch: [%d][%d/%d]\tTime %.3f\ttripErr %.2e'):format(
            epoch, batchNumber, opt.epochSize, timer:time().real, err))
      timer:reset()
      triplet_loss = triplet_loss + err

      -- Return early if the epoch is over.
      if batchNumber >= opt.epochSize then
         return
      end

      -- Return early if the loss is 0 for `numZeros` iterations.
      zeroCounts[zeroIdx] = (err == 0.0) and 1 or 0 -- Boolean to int.
      zeroIdx = (zeroIdx % numZeros) + 1
      if zeroCounts:sum() == numZeros then
         return
      end

      -- Return early if the loss shrinks too much.
      -- if firstLoss == nil then
      --    firstLoss = err
      -- else
      --    -- Triplets trivially satisfied if err=0
      --    if err ~= 0 and firstLoss/err > 4 then
      --       return
      --    end
      -- end

      beginIdx = endIdx + 1
   end
   assert(beginIdx - 1 == numTrips or beginIdx == numTrips)
end
