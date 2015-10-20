"use strict";
// create new svg
var svg = dimple.newSvg("#chart_div", 1000, 600);

// load wine data
d3.csv("data/wineQualityReds.csv", function(data) {

  // create new chart
  var wine_chart = new dimple.chart(svg, data);

  // define x axis
  var x = wine_chart.addMeasureAxis("x", "sulphates");
  x.tickFormat = ".2f";
  x.title = "Sulphates (Potassium Sulphate - g / dm^3)";
  x.overrideMin = 0.5;
  x.overrideMax = 0.9;
  x.fontSize = 12;

  // define y axis
  var y = wine_chart.addMeasureAxis("y", "volatile.acidity");
  y.tickFormat = ".1f";
  y.title = "Volatile Acidity (Acetic Acid - g / dm^3)";
  y.overrideMin = 0.2;
  y.overrideMax = 0.8;
  y.fontSize = 12;

  // add alcohol amount as relative size of bubble on a log scale
  var z = wine_chart.addLogAxis("z", "alcohol", 0.1);

  // plot bubbles, color as quality
  var s = wine_chart.addSeries(["alcohol", "quality"], dimple.plot.bubble);
  s.addOrderRule("quality", true);
  s.aggregate = dimple.aggregateMethod.avg;

  // create legend
  var quality_legend = wine_chart.addLegend(10, 10, 620, 35, "right", [s]);

  // draw visualization
  wine_chart.draw();

  // toggle visibility of bubbles based on score selection
  $(".dimple-legend").click(function(e) {
    var quality = e.currentTarget.firstChild.textContent;
    $(".dimple-bubble.dimple-"+quality).toggle();
  });

  // initially, remove all bubbles
  for (var i = 3; i <= 8; i++) {
    $(".dimple-bubble.dimple-"+i).toggle();
  }

  // initially, make each set of score bubbles load (slowly) before the next
  for (var i = 3; i <= 8; i++) {
    $(".dimple-bubble.dimple-"+i).delay(1000*i).fadeIn(500);
  }

  // get alcohol values
  var alcohol = [];
  for (var i = 0; i < data.length; i++) {
    alcohol.push(parseFloat(data[i]['alcohol']));
  }
  // get stats of alcohol values
  var min = Math.min.apply(null, alcohol);
  var max = Math.max.apply(null, alcohol);
  var sum = alcohol.reduce(function(a,b) {
    return a+b;
  });
  var mean = (sum/alcohol.length).toFixed(1);
  // display stats of alcohol values in summary section
  $("#sizes").html("(min = "+min+", mean = "+mean+", max = "+max+")");

});
      