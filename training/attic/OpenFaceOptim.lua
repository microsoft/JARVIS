-- Modified from https://github.com/facebook/fbnn/blob/master/fbnn/Optim.lua.

local pl = require('pl.import_into')()

local OpenFaceOptim, _ = torch.class('OpenFaceOptim')


-- deepcopy routine that assumes the presence of a 'clone' method in user
-- data should be used to deeply copy. This matches the behavior of Torch
-- tensors.
local function deepcopy(x)
    local typename = type(x)
    if typename == "userdata" then
        return x:clone()
    end
    if typename == "table" then
        local retval = { }
        for k,v in pairs(x) do
            retval[deepcopy(k)] = deepcopy(v)
        end
        return retval
    end
    return x
end

-- Returns weight parameters and bias parameters and associated grad parameters
-- for this module. Annotates the return values with flag marking parameter set
-- as bias parameters set
function OpenFaceOptim.weight_bias_parameters(module)
    local weight_params, bias_params
    if module.weight then
        weight_params = {module.weight, module.gradWeight}
        weight_params.is_bias = false
    end
    if module.bias then
        bias_params = {module.bias, module.gradBias}
        bias_params.is_bias = true
    end
    return {weight_params, bias_params}
end

function OpenFaceOptim:__init(model, optState, checkpoint_data)
    assert(model)
    assert(checkpoint_data or optState)
    assert(not (checkpoint_data and optState))

    self.model = model
    self.modulesToOptState = {}
    -- Keep this around so we update it in setParameters
    self.originalOptState = optState

    -- Each module has some set of parameters and grad parameters. Since
    -- they may be allocated discontinuously, we need separate optState for
    -- each parameter tensor. self.modulesToOptState maps each module to
    -- a lua table of optState clones.
    if not checkpoint_data then
        self.model:apply(function(module)
            self.modulesToOptState[module] = { }
            local params = self.weight_bias_parameters(module)
            -- expects either an empty table or 2 element table, one for weights
            -- and one for biases
            assert(pl.tablex.size(params) == 0 or pl.tablex.size(params) == 2)
            for i, _ in ipairs(params) do
                self.modulesToOptState[module][i] = deepcopy(optState)
                if params[i] and params[i].is_bias then
                    -- never regularize biases
                    self.modulesToOptState[module][i].weightDecay = 0.0
                end
            end
            assert(module)
            assert(self.modulesToOptState[module])
        end)
    else
        local state = checkpoint_data.optim_state
        local modules = {}
        self.model:apply(function(m) table.insert(modules, m) end)
        assert(pl.tablex.compare_no_order(modules, pl.tablex.keys(state)))
        self.modulesToOptState = state
    end
end

local function get_device_for_module(mod)
   local dev_id = nil
   for _, val in pairs(mod) do
      if torch.typename(val) == 'torch.CudaTensor' then
         local this_dev = val:getDevice()
           if this_dev ~= 0 then
               -- _make sure the tensors are allocated consistently
              assert(dev_id == nil or dev_id == this_dev)
               dev_id = this_dev
           end
       end
   end
   return dev_id -- _may still be zero if none are allocated.
end

local function on_device_for_module(mod, f)
   local this_dev = get_device_for_module(mod)
    if this_dev ~= nil then
       return cutorch.withDevice(this_dev, f)
    end
    return f()
end

function OpenFaceOptim:optimizeTriplet(optimMethod, inputs, criterion)
   assert(optimMethod)
   assert(inputs)
   assert(criterion)
   assert(self.modulesToOptState)

   self.model:zeroGradParameters()
   local output = self.model:forward(inputs)

   local err = criterion:forward(output)

   local df_do = criterion:backward(output)
   self.model:backward(inputs, df_do)

    -- We'll set these in the loop that iterates over each module. Get them
    -- out here to be captured.
    local curGrad
    local curParam
    local function fEvalMod(_)
        return err, curGrad
    end

    for curMod, opt in pairs(self.modulesToOptState) do
       on_device_for_module(curMod, function()
                               local curModParams = self.weight_bias_parameters(curMod)
            -- expects either an empty table or 2 element table, one for weights
            -- and one for biases
                               assert(pl.tablex.size(curModParams) == 0 or
                                         pl.tablex.size(curModParams) == 2)
            if curModParams then
               for i, _ in ipairs(curModParams) do
                  if curModParams[i] then
                        -- expect param, gradParam pair
                     curParam, curGrad = table.unpack(curModParams[i])
                     assert(curParam and curGrad)
                     optimMethod(fEvalMod, curParam, opt[i])
                    end
                end
            end
       end)
    end

    return err, output
end
