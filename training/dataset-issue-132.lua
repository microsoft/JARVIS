-- Source: https://github.com/soumith/imagenet-multiGPU.torch/blob/master/dataset.lua
-- Modified by Brandon Amos in Sept 2015 for OpenFace by adding
-- `samplePeople` and `sampleTriplet`.

require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'
tds = require 'tds'

local dataset = torch.class('dataLoader')

local initcheck = argcheck{
   pack=true,
   help=[[
     A dataset class for images in a flat folder structure (folder-name is class-name).
     Optimized for extremely large datasets (upwards of 14 million images).
     Tested only on Linux (as it uses command-line linux utilities to scale up)
]],
   {check=function(paths)
       local out = true;
       for _,v in ipairs(paths) do
          if type(v) ~= 'string' then
             print('paths can only be of string input');
             out = false
          end
       end
       return out
   end,
    name="paths",
    type="table",
    help="Multiple paths of directories with images"},

   {name="sampleSize",
    type="table",
    help="a consistent sample size to resize the images"},

   {name="split",
    type="number",
    help="Percentage of split to go to Training"
   },

   {name="samplingMode",
    type="string",
    help="Sampling mode: random | balanced ",
    default = "balanced"},

   {name="verbose",
    type="boolean",
    help="Verbose mode during initialization",
    default = false},

   {name="loadSize",
    type="table",
    help="a size to load the images to, initially",
    opt = true},

   {name="forceClasses",
    type="table",
    help="If you want this loader to map certain classes to certain indices, "
       .. "pass a classes table that has {classname : classindex} pairs."
       .. " For example: {3 : 'dog', 5 : 'cat'}"
       .. "This function is very useful when you want two loaders to have the same "
    .. "class indices (trainLoader/testLoader for example)",
    opt = true},

   {name="sampleHookTrain",
    type="function",
    help="applied to sample during training(ex: for lighting jitter). "
       .. "It takes the image path as input",
    opt = true},

   {name="sampleHookTest",
    type="function",
    help="applied to sample during testing",
    opt = true},
}

function dataset:__init(...)

   -- argcheck
   local args =  initcheck(...)
   print(args)
   for k,v in pairs(args) do self[k] = v end

   if not self.loadSize then self.loadSize = self.sampleSize; end

   if not self.sampleHookTrain then self.sampleHookTrain = self.defaultSampleHook end
   if not self.sampleHookTest then self.sampleHookTest = self.defaultSampleHook end

   -- find class names
   print('finding class names')
   self.classes = {}
   local classPaths = tds.Hash()
   if self.forceClasses then
      print('Adding forceClasses class names')
      for k,v in pairs(self.forceClasses) do
         self.classes[k] = v
         classPaths[k] = tds.Hash()
      end
   end
   -- loop over each paths folder, get list of unique class names,
   -- also store the directory paths per class
   -- for each class,
   print('Adding all path folders')
   for _,path in ipairs(self.paths) do
      for dirpath in paths.iterdirs(path) do
         dirpath = path .. '/' .. dirpath
         local class = paths.basename(dirpath)
         self.classes[#self.classes + 1] = class
         classPaths[#classPaths + 1] = dirpath
      end
   end

   print(#self.classes .. ' class names found')
   self.classIndices = {}
   for k,v in ipairs(self.classes) do
      self.classIndices[v] = k
   end


   -- find the image path names
   print('Finding path for each image')
   self.imagePath = torch.CharTensor()  -- path to each image in dataset
   self.imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)
   self.classList = {}                  -- index of imageList to each image of a particular class
   self.classListSample = self.classList -- the main list used when sampling data
   local counts = tds.Hash()
   local maxPathLength = 0

   print('Calculating maximum class name length and counting files')
   local length = 0

   local fullPaths = tds.Hash()
   -- iterate over classPaths
   for _,path in pairs(classPaths) do
    local count = 0
      -- iterate over files in the class path
      for f in paths.iterfiles(path) do
        local fullPath = path .. '/' .. f
        maxPathLength = math.max(fullPath:len(), maxPathLength)
        count = count + 1
        length = length + 1
        fullPaths[#fullPaths + 1] = fullPath
      end
      counts[path] = count
   end

   assert(length > 0, "Could not find any image file in the given input paths")
   assert(maxPathLength > 0, "paths of files are length 0?")
   maxPathLength = maxPathLength + 1

   self.imagePath:resize(length, maxPathLength):fill(0)
   local s_data = self.imagePath:data()
   local count = 0
   for _,line in pairs(fullPaths) do
     ffi.copy(s_data, line)
     s_data = s_data + maxPathLength
     if self.verbose and count % 10000 == 0 then
        xlua.progress(count, length)
     end;
     count = count + 1
   end

   self.numSamples = self.imagePath:size(1)
   if self.verbose then print(self.numSamples ..  ' samples found.') end
   --==========================================================================
   print('Updating classList and imageClass appropriately')
   self.imageClass:resize(self.numSamples)
   local runningIndex = 0
   for i=1,#self.classes do
      if self.verbose then xlua.progress(i, #(self.classes)) end
      local clsLength = counts[classPaths[i]]
      if clsLength == 0 then
         error('Class has zero samples: ' .. self.classes[i])
      else
         self.classList[i] = torch.linspace(runningIndex + 1, runningIndex + clsLength, clsLength):long()
         self.imageClass[{{runningIndex + 1, runningIndex + clsLength}}]:fill(i)
      end
      runningIndex = runningIndex + clsLength
   end

   --==========================================================================

   if self.split == 100 then
      self.testIndicesSize = 0
   else
      print('Splitting training and test sets to a ratio of '
               .. self.split .. '/' .. (100-self.split))
      self.classListTrain = {}
      self.classListTest  = {}
      self.classListSample = self.classListTrain
      local totalTestSamples = 0
      -- split the classList into classListTrain and classListTest
      for i=1,#self.classes do
         local list = self.classList[i]
         count = self.classList[i]:size(1)
         local splitidx = math.floor((count * self.split / 100) + 0.5) -- +round
         local perm = torch.randperm(count)
         self.classListTrain[i] = torch.LongTensor(splitidx)
         for j=1,splitidx do
            self.classListTrain[i][j] = list[perm[j]]
         end
         if splitidx == count then -- all samples were allocated to train set
            self.classListTest[i]  = torch.LongTensor()
         else
            self.classListTest[i]  = torch.LongTensor(count-splitidx)
            totalTestSamples = totalTestSamples + self.classListTest[i]:size(1)
            local idx = 1
            for j=splitidx+1,count do
               self.classListTest[i][idx] = list[perm[j]]
               idx = idx + 1
            end
         end
      end
      -- Now combine classListTest into a single tensor
      self.testIndices = torch.LongTensor(totalTestSamples)
      self.testIndicesSize = totalTestSamples
      local tdata = self.testIndices:data()
      local tidx = 0
      for i=1,#self.classes do
         local list = self.classListTest[i]
         if list:dim() ~= 0 then
            local ldata = list:data()
            for j=0,list:size(1)-1 do
               tdata[tidx] = ldata[j]
               tidx = tidx + 1
            end
         end
      end
   end
end

-- size(), size(class)
function dataset:size(class, list)
   list = list or self.classList
   if not class then
      return self.numSamples
   elseif type(class) == 'string' then
      return list[self.classIndices[class]]:size(1)
   elseif type(class) == 'number' then
      return list[class]:size(1)
   end
end

-- size(), size(class)
function dataset:sizeTrain(class)
   if self.split == 0 then
      return 0;
   end
   if class then
      return self:size(class, self.classListTrain)
   else
      return self.numSamples - self.testIndicesSize
   end
end

-- size(), size(class)
function dataset:sizeTest(class)
   if self.split == 100 then
      return 0
   end
   if class then
      return self:size(class, self.classListTest)
   else
      return self.testIndicesSize
   end
end

-- by default, just load the image and return it
function dataset:defaultSampleHook(imgpath)
   local out = image.load(imgpath, 3, 'float')
   out = image.scale(out, self.sampleSize[3], self.sampleSize[2])
   return out
end

-- getByClass
function dataset:getByClass(class)
   local index = math.ceil(torch.uniform() * self.classListSample[class]:nElement())
   local imgpath = ffi.string(torch.data(self.imagePath[self.classListSample[class][index]]))
   return self:sampleHookTrain(imgpath)
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, dataTable, scalarTable)
   local data, scalarLabels, labels
   local quantity = #scalarTable
   local samplesPerDraw
   if dataTable[1]:dim() == 3 then samplesPerDraw = 1
   else samplesPerDraw = dataTable[1]:size(1) end
   if quantity == 1 and samplesPerDraw == 1 then
      data = dataTable[1]
      scalarLabels = scalarTable[1]
      labels = torch.LongTensor(#(self.classes)):fill(-1)
      labels[scalarLabels] = 1
   else
      data = torch.Tensor(quantity * samplesPerDraw,
                          self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
      scalarLabels = torch.LongTensor(quantity * samplesPerDraw)
      labels = torch.LongTensor(quantity * samplesPerDraw, #(self.classes)):fill(-1)
      for i=1,#dataTable do
         local idx = (i-1)*samplesPerDraw
         data[{{idx+1,idx+samplesPerDraw}}]:copy(dataTable[i])
         scalarLabels[{{idx+1,idx+samplesPerDraw}}]:fill(scalarTable[i])
         labels[{{idx+1,idx+samplesPerDraw},{scalarTable[i]}}]:fill(1)
      end
   end
   return data, scalarLabels, labels
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
   if self.split == 0 then
      error('No training mode when split is set to 0')
   end
   quantity = quantity or 1
   local dataTable = tds.Hash()
   local scalarTable = tds.Hash()
   for _=1,quantity do
      local class = torch.random(1, #self.classes)
      local out = self:getByClass(class)
      dataTable[#dataTable + 1] = out
      scalarTable[#scalarTable + 1] = class
   end
   local data, scalarLabels, labels = tableToOutput(self, dataTable, scalarTable)
   return data, scalarLabels, labels
end

-- Naively sample random triplets.
function dataset:sampleTriplet(quantity)
   if self.split == 0 then
      error('No training mode when split is set to 0')
   end
   quantity = quantity or 1
   local dataTable = {}
   local scalarTable = {}

   -- Anchors
   for _=1,quantity do
      local anchorClass = torch.random(1, #self.classes)
      table.insert(dataTable, self:getByClass(anchorClass))
      table.insert(scalarTable, anchorClass)
   end

   -- Positives
   for i=1,quantity do
      local posClass = scalarTable[i]
      table.insert(dataTable, self:getByClass(posClass))
      table.insert(scalarTable, posClass)
   end

   -- Negatives
   for i=1,quantity do
      local posClass = scalarTable[i]
      local negClass = posClass
      while negClass == posClass do
         negClass = torch.random(1, #self.classes)
      end
      table.insert(dataTable, self:getByClass(negClass))
      table.insert(scalarTable, negClass)
   end
   local data, scalarLabels, labels = tableToOutput(self, dataTable, scalarTable)
   return data, scalarLabels, labels
end


function dataset:samplePeople(peoplePerBatch, imagesPerPerson)
   if self.split == 0 then
      error('No training mode when split is set to 0')
   end

   local classes = torch.randperm(#trainLoader.classes)[{{1,peoplePerBatch}}]:int()
   local numPerClass = torch.Tensor(peoplePerBatch)
   for i=1,peoplePerBatch do
      local n = math.min(self.classListSample[classes[i]]:nElement(), imagesPerPerson)
      numPerClass[i] = n
   end

   local data = torch.Tensor(numPerClass:sum(),
                             self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])

   local dataIdx = 1
   for i=1,peoplePerBatch do
      local cls = classes[i]
      local n = numPerClass[i]
      local shuffle = torch.randperm(n)
      for j=1,n do
         imgNum = self.classListSample[cls][shuffle[j]]
         imgPath = ffi.string(torch.data(self.imagePath[imgNum]))
         data[dataIdx] = self:sampleHookTrain(imgPath)
         dataIdx = dataIdx + 1
      end
   end
   assert(dataIdx - 1 == numPerClass:sum())

   return data, numPerClass
end

function dataset:get(i1, i2)
   local indices, quantity
   if type(i1) == 'number' then
      if type(i2) == 'number' then -- range of indices
         indices = torch.range(i1, i2);
         quantity = i2 - i1 + 1;
      else -- single index
         indices = {i1}; quantity = 1
      end
   elseif type(i1) == 'table' then
      indices = i1; quantity = #i1;         -- table
   elseif (type(i1) == 'userdata' and i1:nDimension() == 1) then
      indices = i1; quantity = (#i1)[1];    -- tensor
   else
      error('Unsupported input types: ' .. type(i1) .. ' ' .. type(i2))
   end
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples
   local dataTable = {}
   local scalarTable = {}
   for i=1,quantity do
      -- load the sample
      local idx = self.testIndices[indices[i]]
      local imgpath = ffi.string(torch.data(self.imagePath[idx]))
      local out = self:sampleHookTest(imgpath)
      table.insert(dataTable, out)
      table.insert(scalarTable, self.imageClass[idx])
   end
   local data, scalarLabels, labels = tableToOutput(self, dataTable, scalarTable)
   return data, scalarLabels, labels
end

function dataset:test(quantity)
   if self.split == 100 then
      error('No test mode when you are not splitting the data')
   end
   local i = 1
   local n = self.testIndicesSize
   local qty = quantity or 1
   return function ()
      if i+qty-1 <= n then
         local data, scalarLabelss, labels = self:get(i, i+qty-1)
         i = i + qty
         return data, scalarLabelss, labels
      end
   end
end

return dataset
