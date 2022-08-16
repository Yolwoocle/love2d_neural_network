-- 23:55 https://www.youtube.com/watch?v=hfMk-kjRv4c
local NeuralNetwork = require "neural_network"

random = love.math.random

function love.load()
	love.window.setMode(1000, 800, {resizable = true})
	network = NeuralNetwork:new(2, 10, 10, 2)

	cam_x, cam_y = -220, -220
	learn_iterations = 0

	current_cost = 0

	perf_graph_canvas = love.graphics.newCanvas()
	inited_perf_graph = false
end

function love.update(dt)
	local batch = network:get_datapoint_minibatch()
	cur_batch_len = #batch
	network:learn(batch, 0.1)
	learn_iterations = learn_iterations + 1
end

function love.draw()
	love.graphics.translate(-cam_x, -cam_y)

	-- Draw neural network output
	draw_neural_network_categorization()
	if not inited_perf_graph then
		init_perf_graph(-400, 150)
		inited_perf_graph = true
	end
	visualize_network(220, -200)

	draw_perf_graph(-400, 150)
end

function love.keypressed(key, scancode, is_reapeat)
	if key == "r" then
		network:randomize()
	elseif key == "t" then
		network:tweak_random()
	elseif key == "l" then
		network:learn(network.datapoints, 1)
		learn_iterations = learn_iterations + 1
	end
end

-- VISUALS

function draw_neural_network_categorization()
	-- [[
	local pw = 2 -- pixel width
	local scale = 1

	for ix=-100, 100 do
		for iy=-100, 100 do
			local value = network:classify({ix*scale, iy*scale})
			if value == 1 then
				love.graphics.setColor(1,0,0)
			else
				love.graphics.setColor(0,1,1)
			end
			
			local px, py = ix*pw, iy*pw
			if pw == 1 then
				love.graphics.points(px,py)
			else
				love.graphics.rectangle("fill", px, py, pw,pw)
			end
			
		end
	end
	love.graphics.setColor(1,1,1,1)

	-- Compute number of correct items
	local correct = 0
	local incorrect = 0
	for k,point in pairs(network.datapoints) do
		value = network:classify(point.inputs)
		if point.value == value then
			correct = correct + 1
		else
			incorrect = incorrect + 1
		end
	end
	
	-- Draw datapoints
	for i,point in pairs(network.datapoints) do
		local col = {0.5,0,0}
		if point.value == 2 then
			col = {0,0.5,0.5}
		end
		circle_color(col, "fill", point.inputs[1]*pw, point.inputs[2]*pw, 4)
	end
	--]]
	-- Cost
	current_cost = network:cost(network.datapoints)
	love.graphics.print(concat(
		"FPS: ", love.timer.getFPS(), "\n",
		"Cost: ", current_cost, "\n",
		"Learn Iterations: ", learn_iterations, "\n",
		"Correct: ", correct, " / ", #network.datapoints, " (", 100*correct/#network.datapoints, "%) \n",
		"Incorrect: ", incorrect, " / ", #network.datapoints, " (", 100*incorrect/#network.datapoints, "%) \n",
		"Current batch size: ", cur_batch_len, "\n"
	), -200, 220)
end

function visualize_network(x,y)
	-- if true then return end
	local w = 120
	local h = 50
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
				love.graphics.line(lx, y+i*h, lx+w, y+j*h)
				
				local px = lerp(lx, lx+w, .5)
				local py = lerp(y+i*h, y+j*h, .5)
				local txt = tostring(round(weight,2))
				love.graphics.print(txt, px-get_text_width(txt)/2, py)
			end

			circle_color({1,1,1}, "fill", lx, y+i*h, 5)
		end
		
		--cirlces
		for j=1,#layer.weights[1] do
			circle_color({1,1,1}, "fill", lx+w, y+j*h, 5)
		end
	end
end

function draw_perf_graph(x,y)
	love.graphics.setCanvas(perf_graph_canvas)

		local col
		if current_cost > .5 then
			col = lerp_color({1,1,0},{1,0,0},current_cost*2-1)
		else 
			col = lerp_color({0,1,0},{1,1,0},current_cost*2)
		end
		love.graphics.setColor(col)
		love.graphics.setPointSize(3)
		love.graphics.points(10 + learn_iterations*.2, 100*(1-current_cost))
		love.graphics.setColor(1,1,1,1)
		love.graphics.setPointSize(1)
		
		love.graphics.setCanvas()
		
		love.graphics.draw(perf_graph_canvas,x,y)
	end
	
	function init_perf_graph(x,y,start)
		start = start or 0
		love.graphics.setCanvas(perf_graph_canvas)
		
		love.graphics.setColor(.5,.5,.5)
		love.graphics.line(0, 0, 1000, 0)
		love.graphics.print("0.0", 0, 100)
		love.graphics.line(0, 100, 1000, 100)
		love.graphics.print("1.0", 0, 0)
		love.graphics.line(0, 50, 1000, 50)
		
		love.graphics.setColor(.3,.3,.3)
		for i=2,9,2 do
			love.graphics.line(0, i*10, 1000, i*10)
			love.graphics.print(concat("0.",10-i), 0, i*10)
		end
		
		love.graphics.setColor(1,1,1,1)
		for i=start,start+10000,200 do
			local tx = 10 + i*.2
			love.graphics.line(tx, 100, tx, 120)
			love.graphics.print(tostring(i), tx, 120)
			love.graphics.setColor(.3,.3,.3,1)
			love.graphics.line(tx, 0, tx, 100)
		end
		
	love.graphics.setCanvas()
		
	love.graphics.setColor(1,1,1,1)
	love.graphics.draw(perf_graph_canvas,x,y)
end