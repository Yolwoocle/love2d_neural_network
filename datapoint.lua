local Class = require "class"
require "util"

local Datapoint = Class:inherit()

function Datapoint:init(inputs, value, target_outputs)
    self.inputs = inputs
    self.value = value
    self.target_outputs = target_outputs
end

return Datapoint