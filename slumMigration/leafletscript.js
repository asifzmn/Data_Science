var map = L.map('mapid').setView({lon: 89.412, lat: 23.810}, 6.66);

// add the OpenStreetMap tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://openstreetmap.org/copyright">OpenStreetMap contributors</a>'
}).addTo(map);

L.control.scale().addTo(map);


//var popup = L.popup()
//.setLatLng([21.510, 91.212])
//.setContent("I am a standalone popup.")
//.openOn(map);

//function onMapClick(e) {alert("You clicked the map at " + e.latlng)}
//mymap.on('click', onMapClick);


//var popup = L.popup();

//function onMapClick(e) {
//    popup
//        .setLatLng(e.latlng)
//        .setContent("You clicked the map at " + e.latlng.toString())
//        .openOn(map);
//}

//map.on('click', onMapClick);

function onEachFeature(feature, layer) {
    // does this feature have a property named popupContent?
    if (feature.properties) {

        // layer.on('mouseover', function (e) {
        //     var popup = L.popup();
        //     popup
        //         .setLatLng(e.latlng)
        //         .setContent(feature.properties.DISTNAME)
        //         .openOn(map);
        //
        // });

        layer.on('click', function (e) {

            var popup = L.popup();
            popup
                .setLatLng(e.latlng)
                .setContent(feature.properties.ADM2_EN)
                .openOn(map);

            // Papa.parse('../BD_Slum_Cat.csv',
            Papa.parse('https://raw.githubusercontent.com/asifzmn/Parrot/master/BD_Slum_Cat.csv',
                {
                    download: true,
                    delimiter: ',',
                    header: false,
                    complete: function getRes(results) {
                        csvData = results.data;

                        console.log(csvData)
                        map.removeLayer(theMarker);

                        var alldis = states.features
                        idx = alldis.indexOf(feature);

                        for (i = 0; i < alldis.length; i++) {
                            alldis[i].properties.count = csvData[idx + 1][i + 1]
                        }
                        // console.log(feature.properties.ADM2_EN)


                        theMarker = L.geoJSON(states.features, {
                            onEachFeature: onEachFeature,
                            // filter: function (feature, layer) {
                            //     return feature.properties.show_on_map;
                            // },
                            style: colorSetter
                        });
                        theMarker.addTo(map);
                    }

                });

        });
    }
}

function createColorPalette() {
    return ['#fdd7d4', '#fcccc7', '#fcbcbd', '#fbacb9'
        , '#fa99b3', '#f881aa', '#f767a1', '#eb509c'
        , '#e03a98', '#cd238f', '#b90d84', '#a1017c'
        , '#8b0179', '#740175', '#5f0070', '#49006a']
        ;

    // return ['#fee2bb', '#fdd9a8', '#fdce98', '#fdc38d'
    //     , '#fdb67f', '#fca26d', '#fc8c59', '#f67b51'
    //     , '#f0694a', '#e7533a', '#dc3c28', '#ce2417'
    //     , '#be0f0a', '#ad0000', '#960000', '#7f0000']
    // ;

}

function counts(){
    arr = [1000,2000,3000,4000,
        5000,6000,7000,8000,
        9000,10000,20000,30000,
        40000,50000,100000];
    arr = [0].concat(arr);
    arr = arr.concat([2000000]);
    return arr;
}

function colorSetter(feature, layer) {
    border = "#9a9a9a";
    opacity = 0.666;
    w = 1
    colorPal = createColorPalette()

    for (i = 0; i < 19; i++) {
        if (parseInt(feature.properties.count) === i) {
            return {
                fillOpacity: opacity,
                color: border,
                weight: w,
                fillColor: colorPal[i]
            };
        }
    }
}

function colorDefault(feature, layer) {
    return {color: "#9a9a9a", fillColor: "#ffffff", fillOpacity: 0.9}
}

// var district = new Array(states.features.length)
var alldis = states.features
for (i = 0; i < alldis.length; i++) {
    alldis[i].properties.show_on_map = true
}
// console.log(alldis)

theMarker = L.geoJSON(states.features, {
    onEachFeature: onEachFeature,
    filter: function (feature, layer) {
        return feature.properties.show_on_map;
    },
    style: colorDefault
});
theMarker.addTo(map);

var z = createColorPalette()
console.log(z)



var legend = L.control({ position: "bottomleft" });

legend.onAdd = function(map) {
  colorPal = createColorPalette()
  migCount =  counts()
  var div = L.DomUtil.create("div", "legend");
  div.innerHTML += "<h4>Slum Migrant</h4>";
  for (i = 0; i < colorPal.length; i++)
    div.innerHTML += '<i style="background:'+ colorPal[i] +'"></i><span>'+migCount[i]+'-'+migCount[i+1]+'</span><br>';

  return div;
};

legend.addTo(map);



// console.log(data)
// for (i = 0; i < states.features.length; i++) {
//     theMarker = L.geoJson({"type": "Polygon", "coordinates": states.features[i].geometry.coordinates[0]});
//     theMarker.addTo(map);
//     district[i]=(theMarker);
// }

// map.removeLayer(district[0]);
// map.removeLayer(district[1]);
// map.removeLayer(district[2]);


// L.geoJSON(states, {
// style: function(feature) {
//     switch (feature.properties.party) {
//         case 'Republican': return {color: "#fffb00"};
//         case 'Democrat':   return {color: "#00eaff"};
//     }
// },
// onEachFeature: onEachFeature,
// filter: function(feature, layer) {
//     return feature.properties.show_on_map;
// }
// }).addTo(map);

