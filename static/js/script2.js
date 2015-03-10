"use script";

// Declare variables
// ----------------------------------------------------------------------------------------------------------------------------------------

var screenHeight = $(window).height();
var screenWidth = parseInt($(window).width() * 100) / 100;

var imagesPerRow = 4;
var totalImages = 2 * 4;

var marginWidth = 4;
var availableWidth = screenWidth - (imagesPerRow * 2) * marginWidth;

var Images = [];
var imageObjectOnFocus = undefined;

var __EVENT_ID__ = 0;
var __EVENTS__ = [];


// Start here
// -----------------------------------------------

$(document).ready(function(){
    $("#container").css("width", screenWidth + "px");
    $("#container").css("height", screenHeight + "px");
    for (var i = 0; i < 8; i ++){
        var div = $("<div class = 'image-box'/>");
        $("#container").append(div);
        div.css("width", (parseFloat(availableWidth) / 4 - (1 * marginWidth)) + "px");
        div.css("height", (parseFloat(availableWidth) / 4 - 2) + "px");
        
    }
});