local Class = require "class"
require "util"

local Layer = Class:inherit()

function Layer:init(num_in_nodes, num_out_nodes, weights, biases)
	self.num_in_nodes = num_in_nodes
	self.num_out_nodes = num_out_nodes

	self.weights = table_2d(num_in_nodes, num_out_nodes, 0)
	self.biases = table_1d(num_out_nodes, 0)
    
    self.cost_gradient_w = table_2d(num_in_nodes, num_out_nodes, 0)
	self.cost_gradient_b = table_1d(num_out_nodes, 0)
    self:randomize_weights_and_biases()
end

function Layer:calculate_outputs(inputs)
	local outputs = table_1d(self.num_out_nodes)
	
	for i_out=1, self.num_out_nodes do
		local out = self.biases[i_out]
		for i_in=1, self.num_in_nodes do
			out = out + self.weights[i_in][i_out] * inputs[i_in]
		end
		outputs[i_out] = self:activation_function(out)
	end

	return (outputs)
end

function Layer:randomize_weights_and_biases()
    for i=1, #self.weights do
        for j=1, #self.weights[i] do
            self.weights[i][j] = random_neighbor(1)
        end
    end

    for i=1, #self.biases do
        self.biases[i] = random_neighbor(1)
    end
end

function Layer:tweak_weights_and_biases()
    for i=1, #self.weights do
        for j=1, #self.weights[i] do
            self.weights[i][j] = self.weights[i][j] + random_neighbor(0.05)
        end
    end

    for i=1, #self.biases do
        self.biases[i] = self.biases[i] + random_neighbor(0.05)
    end
end

function Layer:activation_function(x)
    return 1 / (1 + exp(-x))
end

return Layer