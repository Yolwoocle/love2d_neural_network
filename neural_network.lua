local Class = require "class"
local Layer = require "layer"
require "util"

local NeuralNetwork = Class:inherit()

function NeuralNetwork:init(...)
	local nodes_per_layer = {...}
	self.layers = {}
	for i=1, #nodes_per_layer-1 do
		self.layers[i] = Layer:new(nodes_per_layer[i], nodes_per_layer[i+1])
	end

	self.target_data = {}
	for i=1, 50 do
		local x = random_neighbor(50)
		local y = random_neighbor(50)
		local v = (x + y)^3 + cos(x) + sin(y)*300
		local value = (v > 0) and 0 or 1
		table.insert(self.target_data, {
			x=x,
			y=y,
			value=value,
		})
	end
end

function NeuralNetwork:calculate_outputs(inputs)
	for i=1, #self.layers do
		local layer = self.layers[i]
		inputs = layer:calculate_outputs(inputs)
	end
	return inputs
end

function NeuralNetwork:classify(inputs)
	local outputs = self:calculate_outputs(inputs)
	
	local imax = 1
	for i=2, #outputs do
		if outputs[i] > outputs[imax] then
			imax = i
		end
	end
	return imax
end

function NeuralNetwork:randomize()
	for k,layer in pairs(self.layers) do
		layer:randomize_weights_and_biases()
	end
end

function NeuralNetwork:tweak_random()
	for k,layer in pairs(self.layers) do
		layer:tweak_weights_and_biases()
	end
end

return NeuralNetwork