local Class = require "class"
require "util"

local Layer = Class:inherit()

function Layer:init(num_in_nodes, num_out_nodes, weights, biases)
	self.num_in_nodes = num_in_nodes
	self.num_out_nodes = num_out_nodes

	self.weights = table_2d(num_in_nodes, num_out_nodes, 0)
	self.biases = table_1d(num_out_nodes, 0)

    self.inputs = table_1d(num_in_nodes, 0)
    self.weighted_inputs = table_1d(num_out_nodes, 0)
    self.activations = table_1d(num_out_nodes, 0)
    
    self.cost_gradient_w = table_2d(num_in_nodes, num_out_nodes, 0)
	self.cost_gradient_b = table_1d(num_out_nodes, 0)
    self:randomize_weights_and_biases()
end

function Layer:calculate_outputs(inputs)
	local outputs = table_1d(self.num_out_nodes)
    self.inputs = inputs
	
	for i_out=1, self.num_out_nodes do
		local out = self.biases[i_out]
		for i_in=1, self.num_in_nodes do
			out = out + self.weights[i_in][i_out] * inputs[i_in]
		end
		
        local activ = self:activation(out)
        outputs[i_out] = activ

        self.weighted_inputs[i_out] = out
        self.activations[i_out] = activ
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

function Layer:activation(x)
    return 1 / (1 + exp(-x))
end

function Layer:activation_derivative(x)
    local a = self:activation(x)
    return a * (1-a)
end

function Layer:node_cost(output_activ, expected_activ)
    local error = output_activ - expected_activ
    return error * error
end

function Layer:node_cost_derivative(output_activ, expected_activ)
    return 2 * (output_activ - expected_activ)
end

function Layer:calculate_output_layer_node_values(expected_outputs)
    local node_values = table_1d(#expected_outputs, 0)

    for i=1, #expected_outputs do
        -- Node value = dCost/dActivation * dActivation/dWeightedInput
        local cost_derivative = self:node_cost_derivative(self.activations[i], expected_outputs[i])
        local activ_derivative = self:activation_derivative(self.weighted_inputs[i])
        node_values[i] = cost_derivative * activ_derivative
    end

    print("new_node_values", table_to_str(node_values), "expected_outputs", #expected_outputs, table_to_str(expected_outputs))
    return node_values
end

function Layer:update_gradients(node_values)
    for i_out = 1, self.num_out_nodes do
        for i_in = 1, self.num_in_nodes do
            -- This basically tells us how sensitive the cost is to a change of the weight
            print(i_in, self.inputs[i_in], node_values[i_out])
            local derivative_cost_weight = self.inputs[i_in] * node_values[i_out]
            self.cost_gradient_w[i_in][i_out] = self.cost_gradient_w[i_in][i_out] + derivative_cost_weight
            -- ^^^ we add because we thden take the average across all training examples
        end
        local derivative_cost_bias = node_values[i_out]
        self.cost_gradient_b[i_out] = self.cost_gradient_b[i_out] + derivative_cost_bias
    end
end

function Layer:calculate_hidden_layer_node_values(old_layer, old_node_values)
    local new_node_values = table_1d(self.num_out_nodes, 0)

    for i_new = 1, #new_node_values do
        local new_node_value = 0
        for i_old = 1, #old_node_values do
            local weighted_input_derivative = old_layer.weights[i_new][i_old]
            new_node_value = new_node_value + weighted_input_derivative * old_node_values[i_old]
        end
        new_node_value = new_node_value * self:activation_derivative(self.weighted_inputs[i_new])
        new_node_values[i_new] = new_node_value
    end

    return new_node_values
end

return Layer