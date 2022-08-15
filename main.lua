-- 23:55 https://www.youtube.com/watch?v=hfMk-kjRv4c

local weight_1_1, weight_2_1 -- to node 1
local weight_2_1, weight_2_2 -- to node 2

local function classify(input1, input2)
	local out_1 = input1 * weight_1_1 + input1 * weight_2_1
	local out_2 = input2 * weight_2_1 + input2 * weight_2_2

	if out_1 > out_2 then
		return true
	end
	return false
end

function love.load()
	
end

function love.update(dt)

end

function love.draw()
	
end