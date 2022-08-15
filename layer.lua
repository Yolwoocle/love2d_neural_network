local Class = require "class"

local Layer = Class:inherit()

function Layer:init(num_in_nodes)
    self.num_in_nodes = num_in_nodes
    self.num_out_nodes = num_out_nodes
end