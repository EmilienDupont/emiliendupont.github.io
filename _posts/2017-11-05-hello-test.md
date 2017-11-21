---
layout: post
title: Interactive Visualization of Optimization Algorithms in Deep Learning
comments: true
---

![placeholder](https://github.com/EmilienDupont/emiliendupont.github.io/blob/master/_assets/profile.png)

{{ site.url }}/_assets/profile.png

<img src="{{ site.url }}/_assets/profile.png" />

Optimization paths on a simple function.


<style>
.sgd {
    stroke: black;
}

.momentum {
    stroke: blue;
}

.rmsprop {
    stroke: red;
}

.adam {
    stroke: green;
}

.SGD {
    fill: black;
}

.Momentum {
    fill: blue;
}

.RMSProp {
    fill: red;
}

.Adam {
    fill: green;
}

circle:hover {
  fill-opacity: .3;
}
</style>
<div id="optim-viz">

<script src="//d3js.org/d3.v3.min.js"></script>
<script>
var width = 720,
    height = 500,
    nx = 72,
    ny = 50,
    rect_width = parseFloat(width)/nx,
    rect_height = parseFloat(height)/ny,
    drawing_time = 30;

var scale_x = d3.scale.linear()
                      .domain([0, width])
                      .range([0, 1]);

var scale_y = d3.scale.linear()
                      .domain([0, height])
                      .range([height/width, 0]);

var lineFunction = d3.svg.line()
                         .x(function(d) { return d.x; })
                         .y(function(d) { return d.y; })
                         .interpolate("linear");

var color_scale = d3.scale.linear()
      .domain([0, 1])
      .range(['white', 'black'])
    //.domain([0, 0.33, .66, 1])
    //.range(["yellow", "orange", "brown", "purple"]);

var svg = d3.select("#optim-viz")
              .append("svg")
              .attr("width", width)
              .attr("height", height);

var function_g = svg.append("g").on("mousedown", mousedown),
    gradient_path_g = svg.append("g"),
    menu_g = svg.append("g");

// Set up the buttons
var draw_bool = {"SGD" : true, "Momentum" : true, "RMSProp" : true, "Adam" : true};

var buttons = ["SGD", "Momentum", "RMSProp", "Adam"];

menu_g.append("rect").attr("x", 0).attr("y", height - 40).attr("width", width).attr("height", 40).attr("fill", "white").attr("opacity", 0.1);

menu_g.selectAll("circle")
        .data(buttons)
        .enter()
        .append("circle")
        .attr("cx", function(d,i) { return width/4 * (i + 0.25);} )
        .attr("cy", height - 20)
        .attr("r", 10)
        .attr("stroke-width", 0.5)
        .attr("stroke", "black")
        .attr("class", function(d) { console.log(d); return d;})
        .attr("fill-opacity", 0.5)
        .attr("stroke-opacity", 1)
        .on("mousedown", button_press);

menu_g.selectAll("text")
        .data(buttons)
        .enter()
        .append("text")
        .attr("x", function(d,i) { return width/4 * (i + 0.25) + 18;} )
        .attr("y", height - 14)
        .text(function(d) { return d; })
        .attr("text-anchor", "start")
        .attr("font-family", "Helvetica Neue")
        .attr("font-size", 15)
        .attr("font-weight", 200)
        .attr("fill", "white")
        .attr("fill-opacity", 1);

function button_press() {
  var type = d3.select(this).attr("class")
  if (draw_bool[type]) {
    d3.select(this).attr("fill-opacity", 0);
    draw_bool[type] = false;
  } else {
    d3.select(this).attr("fill-opacity", 0.5)
    draw_bool[type] = true;
  }
}

// Set up the function and gradients

// Function params
var params_0 = {'a' : 3, 'delta' : .48},
    params_1 = {'a' : 6, 'delta' : .25},
    params_2 = { 'a' : -2, 'a_x' : -80, 'a_y' : -80, 'delta_x': .3, 'delta_y': .3 },
    params_3 = { 'a' : -1.2, 'a_x' : -80, 'a_y' : -80, 'delta_x': .7, 'delta_y': .2 };

function exp_squared(x, y, params) {
    return params.a * Math.exp( params.a_x * Math.pow(x - params.delta_x, 2) + params.a_y * Math.pow(y - params.delta_y, 2));
}

function exp_squared_grad_x(x, y, params) {
    return 2 * params.a_x * (x - params.delta_x) * exp_squared(x, y, params);
}

function exp_squared_grad_y(x, y, params) {
    return 2 * params.a_y * (y - params.delta_y) * exp_squared(x, y, params);
}

function x_squared(x, params) {
    return params.a * Math.pow(x - params.delta, 2);
}

function x_squared_grad(x, params) {
    return params.a * 2 * (x - params.delta);
}

function f(x, y) {
    var parabolas = x_squared(x, params_0) + x_squared(y, params_1)
        bells = exp_squared(x,y,params_2) + exp_squared(x,y,params_3);
    return parabolas + bells;
}

// Returns gradient of f at (x,y)
function grad_f(x,y) {
    var grad_x = x_squared_grad(x, params_0),
        grad_y = x_squared_grad(y, params_1);
    grad_x += exp_squared_grad_x(x,y,params_2) + exp_squared_grad_x(x,y,params_3);
    grad_y += exp_squared_grad_y(x,y,params_2) + exp_squared_grad_y(x,y,params_3);
    return [grad_x, grad_y];
}

// Returns nx by ny grid of f(x,y) values as a 1 dimensional array.
//   Each entry of array is [x, y, f(x,y)]
function get_f_grid(nx, ny) {
    var grid = []
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            var x = scale_x( parseFloat(i) / nx * width ),
                y = scale_y( parseFloat(j) / ny * height );
            grid.push([x, y, f(x,y)]);
        }
    }
    return grid;
}

// Return min and max of f
function min_max_f(f_array) {
    var min = Infinity,
        max = -Infinity;
    for (i = 0; i < f_array.length; i++) {
        if (f_array[i][2] < min) {
            min = f_array[i][2];
        }
        if (f_array[i][2] > max) {
            max = f_array[i][2];
        }
    }
    return [min, max];
}


// Set up the heatmap

var f_grid = get_f_grid(nx, ny);

var min_max = min_max_f(f_grid);

var min_f = min_max[0],
    max_f = min_max[1];

// Set up function values
function_g.selectAll("rect")
          .data(f_grid)
          .enter()
          .append("rect")
          .attr("x", function(d) { return scale_x.invert(d[0]); })
          .attr("y", function(d) { return scale_y.invert(d[1]); })
          .attr("width", rect_width)
          .attr("height", rect_height)
          .attr("fill", function(d) { return color_scale((d[2] - min_f)/(max_f - min_f)); });


// Set up optimization/gradient descent functions.
// SGD, Momentum, RMSProp, Adam.

function get_sgd_path(x0, y0, learning_rate, num_steps) {
    var sgd_history = [{"x": scale_x.invert(x0), "y": scale_y.invert(y0)}];
    var x1, y1, gradient;
    for (i = 0; i < num_steps; i++) {
        gradient = grad_f(x0, y0);
        x1 = x0 - learning_rate * gradient[0]
        y1 = y0 - learning_rate * gradient[1]
        sgd_history.push({"x" : scale_x.invert(x1), "y" : scale_y.invert(y1)})
        x0 = x1
        y0 = y1
    }
    return sgd_history;
}

function get_momentum_path(x0, y0, learning_rate, num_steps, momentum) {
    var v_x = 0,
        v_y = 0;
    var momentum_history = [{"x": scale_x.invert(x0), "y": scale_y.invert(y0)}];
    var x1, y1, gradient;
    for (i=0; i < num_steps; i++) {
        gradient = grad_f(x0, y0)
        v_x = momentum * v_x - learning_rate * gradient[0]
        v_y = momentum * v_y - learning_rate * gradient[1]
        x1 = x0 + v_x
        y1 = y0 + v_y
        momentum_history.push({"x" : scale_x.invert(x1), "y" : scale_y.invert(y1)})
        x0 = x1
        y0 = y1
    }
    return momentum_history
}

function get_rmsprop_path(x0, y0, learning_rate, num_steps, decay_rate, eps) {
    var cache_x = 0,
        cache_y = 0;
    var rmsprop_history = [{"x": scale_x.invert(x0), "y": scale_y.invert(y0)}];
    var x1, y1, gradient;
    for (i = 0; i < num_steps; i++) {
        gradient = grad_f(x0, y0)
        cache_x = decay_rate * cache_x + (1 - decay_rate) * gradient[0] * gradient[0]
        cache_y = decay_rate * cache_y + (1 - decay_rate) * gradient[1] * gradient[1]
        x1 = x0 - learning_rate * gradient[0] / (Math.sqrt(cache_x) + eps)
        y1 = y0 - learning_rate * gradient[1] / (Math.sqrt(cache_y) + eps)
        rmsprop_history.push({"x" : scale_x.invert(x1), "y" : scale_y.invert(y1)})
        x0 = x1
        y0 = y1
    }
    return rmsprop_history;
}

function get_adam_path(x0, y0, learning_rate, num_steps, beta_1, beta_2, eps) {
    var m_x = 0,
        m_y = 0,
        v_x = 0,
        v_y = 0;
    var adam_history = [{"x": scale_x.invert(x0), "y": scale_y.invert(y0)}];
    var x1, y1, gradient;
    for (i = 0; i < num_steps; i++) {
        gradient = grad_f(x0, y0)
        m_x = beta_1 * m_x + (1 - beta_1) * gradient[0]
        m_y = beta_1 * m_y + (1 - beta_1) * gradient[1]
        v_x = beta_2 * v_x + (1 - beta_2) * gradient[0] * gradient[0]
        v_y = beta_2 * v_y + (1 - beta_2) * gradient[1] * gradient[1]
        x1 = x0 - learning_rate * m_x / (Math.sqrt(v_x) + eps)
        y1 = y0 - learning_rate * m_y / (Math.sqrt(v_y) + eps)
        adam_history.push({"x" : scale_x.invert(x1), "y" : scale_y.invert(y1)})
        x0 = x1
        y0 = y1
    }
    return adam_history;
}

// Functions necessary for path visualizations

function draw_path(path_data, type) {
    var gradient_path = gradient_path_g.selectAll(type)
                        .data(path_data)
                        .enter()
                        .append("path")
                        .attr("d", lineFunction(path_data.slice(0,1)))
                        .attr("class", type)
                        .attr("stroke-width", 3)
                        .attr("fill", "none")
                        .attr("stroke-opacity", 0.5)
                        .transition()
                        .duration(drawing_time)
                        .delay(function(d,i) { return drawing_time * i; })
                        .attr("d", function(d,i) { return lineFunction(path_data.slice(0,i+1));})
                        .remove();

    gradient_path_g.append("path")
                   .attr("d", lineFunction(path_data))
                   .attr("class", type)
                   .attr("stroke-width", 3)
                   .attr("fill", "none")
                   .attr("stroke-opacity", 0.5)
                   .attr("stroke-opacity", 0)
                   .transition()
                   .duration(path_data.length * drawing_time)
                   .attr("stroke-opacity", 0.5);
}

// Start minimization from click on heatmap

function mousedown() {
    // Get initial point
    var point = d3.mouse(this);
    // Minimize and draw paths
    minimize(scale_x(point[0]), scale_y(point[1]));
}

function minimize(x0,y0) {
    gradient_path_g.selectAll("path").remove();

    if (draw_bool.SGD) {
        var sgd_data = get_sgd_path(x0, y0, 1e-3, 500);
        draw_path(sgd_data, "sgd");
    }
    if (draw_bool.Momentum) {
        var momentum_data = get_momentum_path(x0, y0, 1e-3, 200, 0.8);
        draw_path(momentum_data, "momentum");
    }
    if (draw_bool.RMSProp) {
        var rmsprop_data = get_rmsprop_path(x0, y0, 1e-3, 300, 0.99, 1e-6);
        draw_path(rmsprop_data, "rmsprop");
    }
    if (draw_bool.Adam) {
        var adam_data = get_adam_path(x0, y0, 1e-3, 100, 0.7, 0.999, 1e-6);
        draw_path(adam_data, "adam");
    }
}
</script>

</div>


To modify the function or add more optimization algorithms, have a look at the [code](https://bl.ocks.org/EmilienDupont/aaf429be5705b219aaaf8d691e27ca87).


It would be interesting to visualize newer optimization algorithms such as Eve
or YellowFin as well.
