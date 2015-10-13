"use strict";

var svg = dimple.newSvg("#chart_div", 1000, 600);

d3.csv("data/wineQualityReds.csv", function(data) {

  var wine_chart = new dimple.chart(svg, data);

  var x = wine_chart.addMeasureAxis("x", "sulphates");
  x.tickFormat = ".2f";
  x.title = "Sulphates (Potassium Sulphate - g / dm^3)";
  x.overrideMin = 0.5;
  x.overrideMax = 0.9;

  var y = wine_chart.addMeasureAxis("y", "volatile.acidity");
  y.tickFormat = ".1f";
  y.title = "Volatile Acidity (Acetic Acid - g / dm^3)";
  y.overrideMin = 0.2;
  y.overrideMax = 0.8;

  var z = wine_chart.addLogAxis("z", "alcohol", 0.1);

  var s = wine_chart.addSeries(["alcohol", "quality"], dimple.plot.bubble);
  s.addOrderRule("quality", true);
  s.aggregate = dimple.aggregateMethod.avg;

  var quality_legend = wine_chart.addLegend(10, 10, 620, 35, "right", [s]);

  wine_chart.draw();

  $('p').append('<br /><small>(click to add/remove)</small>');

  $(".dimple-legend").click(function(e) {
    var quality = e.currentTarget.firstChild.textContent;
    $(".dimple-bubble.dimple-"+quality).toggle();
  });

  var alcohol = [];
  for (var i = 0; i < data.length; i++) {
    alcohol.push(parseFloat(data[i]['alcohol']));
  }

  var min = Math.min.apply(null, alcohol);
  var max = Math.max.apply(null, alcohol);
  var sum = alcohol.reduce(function(a,b) {
    return a+b;
  });
  var mean = (sum/alcohol.length).toFixed(1);

  $("#sizes").html("(min:"+min+", mean:"+mean+", max:"+max+")");

});
      