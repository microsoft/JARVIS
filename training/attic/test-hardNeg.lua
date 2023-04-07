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

testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local batchNumber
local triplet_loss
local timer = torch.Timer()

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   model:evaluate()
   model:cuda()

   triplet_loss = 0
   local i = 1
   while batchNumber < opt.epochSize do
      donkeys:addjob(
         function()
            local inputs, numPerClass = trainLoader:samplePeople(opt.peoplePerBatch,
                                                                 opt.imagesPerPerson)
            inputs = inputs:float()
            numPerClass = numPerClass:float()
            return sendTensor(inputs), sendTensor(numPerClass)
         end,
         testBatch
      )
      if i % 5 == 0 then
         donkeys:synchronize()
         collectgarbage()
      end
      i = i + 1
   end

   donkeys:synchronize()
   cutorch.synchronize()

   triplet_loss = triplet_loss / opt.testEpochSize
   testLogger:add{
      ['avg triplet loss (test set)'] = triplet_loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average triplet loss (per batch): %.2f',
                       epoch, timer:time().real, triplet_loss))
   print('\n')


end

local inputsCPU = torch.FloatTensor()
local numPerClass = torch.FloatTensor()

function testBatch(inputsThread, numPerClassThread)
   if batchNumber >= opt.epochSize then
      return
   end

   cutorch.synchronize()
   timer:reset()
   receiveTensor(inputsThread, inputsCPU)
   receiveTensor(numPerClassThread, numPerClass)

   -- inputs:resize(inputsCPU:size()):copy(inputsCPU)

   local numImages = inputsCPU:size(1)
   local embeddings = torch.Tensor(numImages, 128)
   local singleNet = model.modules[1]
   local beginIdx = 1
   local inputs = torch.CudaTensor()
   while beginIdx <= numImages do
      local endIdx = math.min(beginIdx+opt.testBatchSize-1, numImages)
      local range = {{beginIdx,endIdx}}
      local sz = inputsCPU[range]:size()
      inputs:resize(sz):copy(inputsCPU[range])
      local reps = singleNet:forward(inputs):float()
      embeddings[range] = reps

      beginIdx = endIdx + 1
   end
   assert(beginIdx - 1 == numImages)

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
      local endIdx = math.min(beginIdx+opt.testBatchSize, numTrips)

      local range = {{beginIdx,endIdx}}
      local sz = as[range]:size()
      asCuda:resize(sz):copy(as[range])
      psCuda:resize(sz):copy(ps[range])
      nsCuda:resize(sz):copy(ns[range])

      local output = model:forward({asCuda, psCuda, nsCuda})
      local err = criterion:forward(output)

      cutorch.synchronize()
      batchNumber = batchNumber + 1
      print(string.format('Epoch: [%d][%d/%d] Triplet Loss: %.2f',
                          epoch, batchNumber, opt.testEpochSize, err))
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
