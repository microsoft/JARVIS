-- Source: https://github.com/soumith/imagenet-multiGPU.torch/blob/master/dataset.lua
-- Modified by Brandon Amos in Sept 2015 for OpenFace by adding
-- `samplePeople` and `sampleTriplet`.

require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local dir = require 'pl.dir'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'

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
   self.classes = {}
   local classPaths = {}
   if self.forceClasses then
      for k,v in pairs(self.forceClasses) do
         self.classes[k] = v
         classPaths[k] = {}
      end
   end
   local function tableFind(t, o) for k,v in pairs(t) do if v == o then return k end end end
   -- loop over each paths folder, get list of unique class names,
   -- also store the directory paths per class
   -- for each class,
   for _,path in ipairs(self.paths) do
      local dirs = dir.getdirectories(path);
      for _,dirpath in ipairs(dirs) do
         local class = paths.basename(dirpath)
         local idx = tableFind(self.classes, class)
         if not idx then
            table.insert(self.classes, class)
            idx = #self.classes
            classPaths[idx] = {}
         end
         if not tableFind(classPaths[idx], dirpath) then
            table.insert(classPaths[idx], dirpath);
         end
      end
   end

   self.classIndices = {}
   for k,v in ipairs(self.classes) do
      self.classIndices[v] = k
   end

   -- define command-line tools, try your best to maintain OSX compatibility
   local wc = 'wc'
   local cut = 'cut'
   local find = 'find'
   if jit.os == 'OSX' then
      wc = 'gwc'
      cut = 'gcut'
      find = 'gfind'
   end
   ----------------------------------------------------------------------
   -- Options for the GNU find command
   local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- find the image path names
   self.imagePath = torch.CharTensor()  -- path to each image in dataset
   self.imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)
   self.classList = {}                  -- index of imageList to each image of a particular class
   self.classListSample = self.classList -- the main list used when sampling data

   print('running "find" on each class directory, and concatenate all'
         .. ' those filenames into a single file containing all image paths for a given class')
   -- so, generates one file per class
   local classFindFiles = {}
   for i=1,#self.classes do
      classFindFiles[i] = os.tmpname()
   end
   local combinedFindList = os.tmpname();

   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- iterate over classes
   for i, _ in ipairs(self.classes) do
      -- iterate over classPaths
      for _,path in ipairs(classPaths[i]) do
         local command = find .. ' "' .. path .. '" ' .. findOptions
            .. ' >>"' .. classFindFiles[i] .. '" \n'
         tmphandle:write(command)
      end
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)

   print('now combine all the files to a single large file')
   tmpfile = os.tmpname()
   tmphandle = assert(io.open(tmpfile, 'w'))
   -- concat all finds to a single large file in the order of self.classes
   for i=1,#self.classes do
      local command = 'cat "' .. classFindFiles[i] .. '" >>' .. combinedFindList .. ' \n'
      tmphandle:write(command)
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)

   --==========================================================================
   print('load the large concatenated list of sample paths to self.imagePath')
   local maxPathLength = tonumber(sys.fexecute(wc .. " -L '"
                                                  .. combinedFindList .. "' |"
                                                  .. cut .. " -f1 -d' '")) + 1
   local length = tonumber(sys.fexecute(wc .. " -l '"
                                           .. combinedFindList .. "' |"
                                           .. cut .. " -f1 -d' '"))
   assert(length > 0, "Could not find any image file in the given input paths")
   assert(maxPathLength > 0, "paths of files are length 0?")
   self.imagePath:resize(length, maxPathLength):fill(0)
   local s_data = self.imagePath:data()
   local count = 0
   for line in io.lines(combinedFindList) do
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
      local clsLength = tonumber(sys.fexecute(wc .. " -l '"
                                              .. classFindFiles[i] .. "' |"
                                              .. cut .. " -f1 -d' '"))
      if clsLength == 0 then
         error('Class has zero samples: ' .. self.classes[i])
      else
         -- self.classList[i] = torch.linspace(runningIndex + 1, runningIndex + clsLength, clsLength):long()
         self.classList[i] = torch.range(runningIndex + 1, runningIndex + clsLength):long()
         self.imageClass[{{runningIndex + 1, runningIndex + clsLength}}]:fill(i)
      end
      runningIndex = runningIndex + clsLength
   end

   --==========================================================================
   -- clean up temporary files
   print('Cleaning up temporary files')
   local tmpfilelistall = ''
   for i=1,#(classFindFiles) do
      tmpfilelistall = tmpfilelistall .. ' "' .. classFindFiles[i] .. '"'
      if i % 1000 == 0 then
         os.execute('rm -f ' .. tmpfilelistall)
         tmpfilelistall = ''
      end
   end
   os.execute('rm -f '  .. tmpfilelistall)
   os.execute('rm -f "' .. combinedFindList .. '"')
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
   local out = image.load(imgpath, 3, byte)
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
   local dataTable = {}
   local scalarTable = {}
   for _=1,quantity do
      local class = torch.random(1, #self.classes)
      local out = self:getByClass(class)
      table.insert(dataTable, out)
      table.insert(scalarTable, class)
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
   local nSamplesPerClass = torch.Tensor(peoplePerBatch)
   for i=1,peoplePerBatch do
      local nSample = math.min(self.classListSample[classes[i]]:nElement(), imagesPerPerson)
      nSamplesPerClass[i] = nSample
   end

   local data = torch.Tensor(nSamplesPerClass:sum(),
                             self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])

   local dataIdx = 1
   for i=1,peoplePerBatch do
      local cls = classes[i]
      local nSamples = nSamplesPerClass[i]
      local nTotal = self.classListSample[classes[i]]:nElement()
      local shuffle = torch.randperm(nTotal)
      for j = 1, nSamples do
         imgNum = self.classListSample[cls][shuffle[j]]
         imgPath = ffi.string(torch.data(self.imagePath[imgNum]))
         data[dataIdx] = self:sampleHookTrain(imgPath)
         dataIdx = dataIdx + 1
      end
   end
   assert(dataIdx - 1 == nSamplesPerClass:sum())

   return data, nSamplesPerClass
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
