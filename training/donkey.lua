-- Source: https://github.com/soumith/imagenet-multiGPU.torch/blob/master/donkey.lua
--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local gm = assert(require 'graphicsmagick')
paths.dofile('dataset.lua')
paths.dofile('util.lua')
ffi=require 'ffi'

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(opt.cache, 'trainCache.t7')
-- local testCache = paths.concat(opt.cache, 'testCache.t7')

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = {3, opt.imgDim, opt.imgDim}
local sampleSize = {3, opt.imgDim, opt.imgDim}

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   -- load image with size hints
   local input = gm.Image():load(path, self.loadSize[3], self.loadSize[2])

   input:size(self.sampleSize[3], self.sampleSize[2])

   local out = input

   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out:flop(); end

   out = out:toTensor('float','RGB','DHW')

   return out
end

if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(opt.data)},
      loadSize = loadSize,
      sampleSize = sampleSize,
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")

end

-- End of train loader section
--------------------------------------------------------------------------------

--[[ Section 2: Create a test data loader (testLoader), ]]--
-- if opt.testEpochSize > 0 then
--   if paths.filep(testCache) then
--     print('Loading test metadata from cache')
--     testLoader = torch.load(testCache)
--   else
--     print('Creating test metadata')
--     testLoader = dataLoader{
--       paths = {paths.concat(opt.data, 'val')},
--       loadSize = loadSize,
--       sampleSize = sampleSize,
--       -- split = 0,
--       split = 100,
--       verbose = true,
--       -- force consistent class indices between trainLoader and testLoader
--       forceClasses = trainLoader.classes
--     }
--     torch.save(testCache, testLoader)
--   end
--   collectgarbage()
-- end
-- -- End of test loader section
