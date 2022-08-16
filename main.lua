-- 23:55 https://www.youtube.com/watch?v=hfMk-kjRv4c
local NeuralNetwork = require "neural_network"

random = love.math.random

local function visualize_network(x,y)
	local w = 60
	for k,layer in pairs(network.layers) do
		local lx = x + (k-1) * w
		for i=1,#layer.weights do
			for j=1,#layer.weights[i] do
				local weight = layer.weights[i][j]
				local col
				if weight > 0 then
					col = lerp_color({1,1,1}, {0,1,0}, weight)
				else
					col = lerp_color({1,1,1}, {1,0,0}, abs(weight))
				end
				love.graphics.setColor(col)
				love.graphics.line(lx, y+i*w, lx+w, y+j*w)
				
				local px = lerp(lx, lx+w, .5)
				local py = lerp(y+i*w, y+j*w, .5)
				local txt = tostring(round(weight,2))
				love.graphics.print(txt, px-get_text_width(txt)/2, py)
			end

			circle_color({1,1,1}, "fill", lx, y+i*w, 5)
		end
		
		--cirlces
		for j=1,#layer.weights[1] do
			circle_color({1,1,1}, "fill", lx+w, y+j*w, 5)
		end
	end
end

function love.load()
	network = NeuralNetwork:new(2, 5, 5, 2)
end

function love.update(dt)

end

function love.draw()
	local pw = 4
	for ix=0, 100 do
		for iy=0, 100 do
			local value = network:classify({ix-50, iy-50})
			if value == 1 then
				love.graphics.setColor(1,0,0)
			else
				love.graphics.setColor(0,1,1)
			end
			love.graphics.rectangle("fill", ix*pw,iy*pw, pw,pw)
			
		end
	end
	love.graphics.setColor(1,1,1,1)

	visualize_network(450, 20)
	
	love.graphics.print(love.timer.getFPS(), 0, love.graphics.getHeight()-30)
end

function love.keypressed(key, scancode, is_reapeat)
	if key == "r" then
		network:randomize()
	end
	if key == "t" then
		network:tweak_random()
	end
end
