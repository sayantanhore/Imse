// Holds the utility functions

"use strict";

var IMG_GAL = IMG_GAL || {};

IMG_GAL.util = {
    removeUnit: function(measure){
        //return measure.replace("px", "");
        
        return parseFloat(measure.substr(0, measure.length - 2));
    },
    attachUnit: function(measure){
        return (measure + 'px');
    },
    
    getPxForEm: function(x){
        var multiplyWith = this.removeUnit($('body').css("font-size"));
        return x * multiplyWith;
    }
};