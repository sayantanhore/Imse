"use strict";

var IMG_GAL = IMG_GAL || {};

IMG_GAL.globals = (function(){
    
    var loc = window.location.pathname.substr(0, window.location.pathname.indexOf("/start"));
    
    var screenHeight = $(window).height();
    var screenWidth = parseFloat($(window).width() * 100) / 100;
    
    var rowHeightDivideFactor = 3.5;
    var availableWidth = 0;
    
    var images = [];
    var numberOfImagesPerIteration = 12;
    
    return {
                    getScreenHeight: function(){
                        return screenHeight;
                    },
                    getScreenWidth: function(){
                        return screenWidth;
                    },
                    getRowHeightDivideFactor: function(){
                        return rowHeightDivideFactor;
                    },
                    setAvailableWidth: function(){
                        
                    },
                    getAvailableWidth: function(){
                        return availableWidth;
                    },
                    resetImages: function(){
                        images = [];
                    },
                    addNewImage: function(imgModelObj){
                        images.push(imgModelObj);
                    },
                    getImages: function(){
                        return images;
                    },
                    getNumberOfImagesPerIteration: function(){
                        return numberOfImagesPerIteration;
                    },
                    getLoc: function(){
                        return loc;
                    }
                }
})();
