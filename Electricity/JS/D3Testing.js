// set the dimensions and margins of the graph
var margin = {top: 10, right: 30, bottom: 30, left: 60},
    width = 460 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

// append the svg object to the body of the page
var svg = d3.select("#my_dataviz")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");

//Read the data
d3.csv("https://raw.githubusercontent.com/holtzy/data_to_viz/master/Example_dataset/2_TwoNum.csv", function (data) {

    // Add X axis
    var x = d3.scaleLinear()
        .domain([0, 4000])
        .range([0, width]);
    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x));

    // Add Y axis
    var y = d3.scaleLinear()
        .domain([0, 500000])
        .range([height, 0]);
    svg.append("g")
        .call(d3.axisLeft(y));

    // Add dots
    svg.append('g')
        .selectAll("dot")
        .data(data)
        .enter()
        .append("circle")
        .attr("cx", function (d) {
            return x(d.GrLivArea);
        })
        .attr("cy", function (d) {
            return y(d.SalePrice);
        })
        .attr("r", 1.5)
        .style("fill", "#69a8b3")

    var colors=
    ['#F70D1A', '#2B3856', '#C35817', '#B2C248',
        '#3BB9FF', '#893BFF', '#7D0541', '#F3E5AB',
        '#C12869', '#BCC6CC', '#8AFB17', '#348781',
        '#C7A317', '#EDE275', '#C6AEC7']



})
