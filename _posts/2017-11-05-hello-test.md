---
layout: post
title: Interactive Visualization of Optimization Algorithms in Deep Learning
comments: true
---

Here is a new blog post.
Can you include javascript in a blog post? Let's try to!
Does _this_ and **this** work?

<div id="rain">

<script src="//d3js.org/d3.v3.min.js"></script>
<script>
var width = 960,
    height = 500;

var svg = d3.select("#rain")
              .append("svg")
              .attr("width", width)
              .attr("height", height);

function raindrop(size, duration, delay, x_pos, y_pos) {
    var drop = svg.append("circle")
            .attr("cx", x_pos)
            .attr("cy", y_pos)
            .attr("r", 0)
            .attr("stroke", "#5FC3E4")
            .attr("stroke-width", 2)
            .attr("fill", "none")
            .attr("opacity", 1);

    drop.transition()
       .delay(delay)
       .duration(duration)
       .attr("r", size)
       .attr("stroke-width", 0)
       .attr("opacity", 0.5)
       .ease("circleout");
}

function make_it_rain(num_drops) {
    console.log("Making it rain");
    for (i = 0; i < num_drops; i++) {
        var size = 50 * Math.random() + 50,
            duration = 50 * Math.random() + 750,
            delay = 5000 * Math.random(),
            x_pos = width * Math.random(),
            y_pos = height * Math.random();
        raindrop(size, duration, delay, x_pos, y_pos);
    }
}

d3.timer(function(elapsed) {
    if (elapsed % 5000 < 50) {
        make_it_rain(50);
    }
})
</script>

</div>
