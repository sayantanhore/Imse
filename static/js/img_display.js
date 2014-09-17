// JS script for IMSE

// Author: Sayantan Hore
// Created: 13.08.2014

"use strict";

var imageBoxWidth;

function setImagesToPlace(img){
    var img_width = img.width();
    var img_height = img.height();
    //console.log(img_width + ", " + img_height)
    if (img_width >= img_height){
        //console.log("Width larger");
        img.width(imageBoxWidth - 5);
    }
    else{
        console.log(imageBoxWidth);
        img.height(imageBoxWidth - 5);
    }
    var marginTop = imageBoxWidth - img.height();
    img.css("margin-top", parseFloat(marginTop / 2) + "px");
}

function displayImages(imageNo){
    var imageContainer = $("<div class='col-xs-12 col-md-12'></div>");
    imageContainer.addClass("image-container");
    for (var i = 0; i < 2; i++){
        imageContainer.append($("<div class = 'thumbnail'></div>"));
    }
    $(".container-fluid").append(imageContainer);
    $(".thumbnail").each(function(index){
        //imageBoxWidth = $(this).width();
        imageBoxWidth = parseFloat($(".image-container").width() / 4.0) - 2;
        $(this).width(imageBoxWidth);
        $(this).height(imageBoxWidth);
        var img = $("<img></img>");
        if (imageNo === 1){
            img.attr("src", "../media/messi.jpg");
        }
        else if (imageNo === 2){
            img.attr("src", "../media/mig29.jpg");
        }
        else if (imageNo === 3){
            img.attr("src", "../media/interaction.jpg");
        }
        
        $(this).append(img);
        
        var slider = $("<input class = 'slider' type='range' min='0.0' max='1.0' value='0.5' step='0.1'>")
        $(this).append(slider);
        slider.width(imageBoxWidth);
        var containerTop = img.css("top");
        console.log((parseInt(index / 4) + 1) * imageBoxWidth * 90 / 100);
        //slider.css("top", (parseInt(index / 4) + 1) * imageBoxWidth  + "px");
        
        
    })
    $(".thumbnail img").load(function(){
        setImagesToPlace($(this));
    });
    
    
    
}