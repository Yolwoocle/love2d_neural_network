local Class = require "class"
local Layer = require "layer"
local Datapoint = require "datapoint"
require "util"

local NeuralNetwork = Class:inherit()

function NeuralNetwork:init(...)
	local nodes_per_layer = {...}
	self.layers = {}
	for i=1, #nodes_per_layer-1 do
		self.layers[i] = Layer:new(nodes_per_layer[i], nodes_per_layer[i+1])
	end

	self.datapoints = {}
	local a = random_range(200, 500)
	local b = random_range(3,6)
	for i=1, 200 do
		local x = random_neighbor(100)
		local y = random_neighbor(100)

		local v = (x*y*.1*.1)^2 - 40
		-- local v = (x+y)^3 - 40
		-- local v = 2
		-- if x<0 and y<0 then
		-- 	v = -1
		-- end
		local value, target_outputs
		if v > 0 then
			value = 1
			target_outputs = {1, 0}
		else
			value = 2
			target_outputs = {0, 1}
		end
		table.insert(self.datapoints, Datapoint:new({x,y}, value, target_outputs))
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

function NeuralNetwork:node_cost(output_activ, target_activ)
    local error = output_activ - target_activ
    return error * error
end

function NeuralNetwork:datapoint_cost(datapoint)
	local outputs = self:calculate_outputs(datapoint.inputs)
	local cost = 0

	for i = 1, #outputs do
		cost = cost + self:node_cost(outputs[i], datapoint.target_outputs[i])
	end

	return cost
end

function NeuralNetwork:cost(data)
	local total = 0

	for k, point in pairs(data) do
		total = total + self:datapoint_cost(point)
	end

	return total / #data
end

function NeuralNetwork:learn(training_data, learn_rate)
    local h = 0.0001
    local old_cost = self:cost(training_data)

	for k, layer in pairs(self.layers) do
		for i_in = 1, layer.num_in_nodes do
			for i_out = 1, layer.num_out_nodes do
				layer.weights[i_in][i_out] = layer.weights[i_in][i_out] + h
				local diff_cost = self:cost(training_data) - old_cost
				layer.weights[i_in][i_out] = layer.weights[i_in][i_out] - h
				layer.cost_gradient_w[i_in][i_out] = diff_cost / h
			end
		end
		
		for i_out = 1, layer.num_out_nodes do
			layer.biases[i_out] = layer.biases[i_out] + h
			local diff_cost = self:cost(training_data) - old_cost
			layer.biases[i_out] = layer.biases[i_out] - h
			layer.cost_gradient_b[i_out] = diff_cost / h
		end
	end

	self:apply_cost_gradients(learn_rate)
end

function NeuralNetwork:apply_cost_gradients(learn_rate)
	for k, layer in pairs(self.layers) do
		for i_out = 1, layer.num_out_nodes do
			layer.biases[i_out] = layer.biases[i_out] - learn_rate * layer.cost_gradient_b[i_out]
			for i_in = 1, layer.num_in_nodes do
				layer.weights[i_in][i_out] = layer.weights[i_in][i_out] - learn_rate * layer.cost_gradient_w[i_in][i_out]
			end
		end
	end
end

function NeuralNetwork:get_datapoint_minibatch()
	local maxskip = 8
	local t = {}
	i = 1
	while i <= #self.datapoints do
		table.insert(t, self.datapoints[i])
		i = i + love.math.random(1, maxskip)
	end
	return t
end

return NeuralNetwork