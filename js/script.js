"use strict";
// create new svg
var svg = dimple.newSvg("#chart_div", 1000, 600);

// load wine data
d3.csv("data/data.csv", function(data) {

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

  // plot bubbles, color as quality
  var s = wine_chart.addSeries(["alcohol", "quality"], dimple.plot.bubble);
  s.addOrderRule("quality", true);
  s.aggregate = dimple.aggregateMethod.avg;

  // create legend
  var quality_legend = wine_chart.addLegend(10, 10, 560, 35, "right", [s]);

  // draw visualization
  wine_chart.draw();

  // initially, remove all bubbles
  for (var i = 0; i <= 2; i++) {
    $(".dimple-bubble.dimple-"+i).toggle();
  }

  // initially, make each set of score bubbles load (slowly) before the next
  for (var i = 0; i <= 2; i++) {
    $(".dimple-bubble.dimple-"+i).delay(1000*i).fadeIn(500);
  }

  $("text.dimple-legend-text:contains('0')").text('low');
  $("text.dimple-legend-text:contains('1')").text('mid');
  $("text.dimple-legend-text:contains('2')").text('high');

  // toggle visibility of bubbles based on score selection
  $(".dimple-legend").click(function(e) {
    var quality = e.currentTarget.firstChild.textContent;

    var scores = { "low":"0", "mid":"1", "high":"2" }

    $(".dimple-bubble.dimple-"+scores[quality]).toggle();
  });


});
      