"use strict";

// Declare variables
// ----------------------------------------------------------------------------------------------------------------------------------------

var screenHeight = $(window).height();
var screenWidth = parseInt($(window).width() * 100) / 100;

var marginWidth = 3;

var containerHeight = 0;
var containerWidth = 0;
var availableWidth = 0;

var availableHeight = screenHeight;
var imageHeightFactor = 3.25;
var totalNoOfImages = 13;

var imagesInCurrentRow = [];

var Images = [];

var __EVENT_ID__ = 0;
var __EVENTS__ = [];


// Update each image onchange location and dimension
// ----------------------------------------------------------------------------------------------------------------------------------------

function updateImage(){
    
}

// Adjust each row after the last image is placed
// ----------------------------------------------------------------------------------------------------------------------------------------

function adjustRow(availableWidth, lastRow){

    var firstImageIndex = imagesInCurrentRow[0];
    var lastImageIndex = imagesInCurrentRow[imagesInCurrentRow.length - 1];
    console.log("Last Image :: " + lastImageIndex);
    console.log($("img").eq(lastImageIndex).width());
    if (lastRow == true){
        $("img").slice(firstImageIndex + 1, lastImageIndex + 1).each(function(){
            $(this).css('margin-left', marginWidth + 'px')
            $(this).css('margin-bottom', marginWidth + 'px')
        });
    }
    else{
        
        var heightToApply = $("img").eq(lastImageIndex).height();
        
        $("img").slice(firstImageIndex, lastImageIndex).height(heightToApply);

        var occupiedWidth = 0;
        $("img").slice(firstImageIndex, lastImageIndex + 1).each(function(){
            occupiedWidth += $(this).width();
        });
        console.log("Occupied width :: " + occupiedWidth);
        console.log("Screen width :: " + screenWidth);
        var heightToIncrease = parseFloat((screenWidth - occupiedWidth - ((imagesInCurrentRow.length + 2) * marginWidth)) * heightToApply) / occupiedWidth;
        $("img").slice(firstImageIndex, lastImageIndex + 1).each(function(){
            //alert("Changing");
            heightToApply = $(this).height() + heightToIncrease;
            var w = $(this).width();
            var h = $(this).height();
            $(this).height(heightToApply);
            $(this).width(parseFloat(heightToApply * w) / h - marginWidth);
        });

        $("img").slice(firstImageIndex + 1, lastImageIndex + 1).each(function(){
            $(this).css('margin-left', marginWidth + 'px');
        });

        //$("img").eq(lastImageIndex).css('margin-right', 1 * marginWidth + 'px');
        
    }
    $("img").slice(firstImageIndex, lastImageIndex + 1).each(function(index){
        console.log(Images[firstImageIndex + index].imageIndex);
        /*
        Images[index].height = $(this).height();
        Images[index].width = $(this).width();
        Images[index].top = $(this).offset().top;
        Images[index].left = $(this).offset().left;
        */
        $(this).trigger('dimensionChanged', firstImageIndex + index);
        
    });
}

// Place each image in a row based on available place left
// ----------------------------------------------------------------------------------------------------------------------------------------

var setImageInPlace = function(containerHeight, containerWidth, availableWidth, index){

    //var imgPath = "../static/images/im" + images[index]+ ".jpg";
    var imgPath = Images[index].imagePath;
    var image = $('<img src = ' + imgPath + ' />');
    
    image.on('click', function(event){
        var target = $(event.target);
        $.each(Images, function(index){
            //alert(target.attr('src'));
            if (Images[index].image.attr('src') === target.attr('src')){
                //alert(true);
                var newImageIndex = -999;
                do{
                    newImageIndex = Math.ceil(Math.random() * 100);
                }while(newImageIndex < totalNoOfImages);
                Images[index].changeImage(newImageIndex);
                
            }
        });
        
    });
    
    image.on('dimensionChanged', function(event, position){
        var target = $(event.target);
        Images[position].dim.height = target.height();
        Images[position].dim.width = target.width();
        Images[position].loc.top = target.offset().top;
        Images[position].loc.left = target.offset().left;
        console.log(' :Height: ' + Images[position].dim.height + ' :Top: ' + Images[position].loc.top + ' :Width: ' + Images[position].dim.width + ' :Left: ' + Images[position].loc.left);
    });
    image.on('load', function(){

        console.log("Image :: " + $(this).attr('src'));
        $(this).height(containerHeight / imageHeightFactor);
        var imgHeight = $(this).height();
        var imgWidth = $(this).width();
        var imgTop = $(this).offset().top;
        var imgLeft = $(this).offset().left;

        console.log("Height :: " + imgHeight);
        console.log("Width :: " + imgWidth);
        console.log("TOP :: " + $(this).offset().top);
        console.log("Left :: " + $(this).offset().left);

        console.log("Availablewidth :: " + availableWidth);


        if (availableWidth < (imgWidth + marginWidth)){
            if (availableWidth >= (Math.ceil(imgWidth * 0.65) + marginWidth)){
                $(this).width(availableWidth - marginWidth);
                console.log("Changed Width :: " + $(this).width());
                $(this).height(imgHeight * $(this).width() / imgWidth);
                console.log("Changed Height :: " + $(this).height());
                availableWidth -= ($(this).width() + marginWidth);
                imagesInCurrentRow.push(index);
            }
            else{
                $(this).css('margin-left', 2 * marginWidth + 'px');
                console.log("Current images :: " + imagesInCurrentRow);

                adjustRow(availableWidth, false);

                imagesInCurrentRow = [];
                imagesInCurrentRow.push(index);
                availableWidth = containerWidth - 2 * marginWidth;
                availableWidth -= ($(this).width() + marginWidth);
            }

        }
        else{
            if (index == 0){
                $(this).css('margin-left', 2 * marginWidth + 'px');
            }
            availableWidth -= ($(this).width() + marginWidth);
            imagesInCurrentRow.push(index);
        }
        
        // Update model
        
        Images[index].image = $(this);
        Images[index].placed = true;
        
        if (index < Images.length - 1){
            setImageInPlace(containerHeight, containerWidth, availableWidth, index + 1);
        }
        else if (index == Images.length - 1){

            if (availableWidth > 0){
                adjustRow(availableWidth, true);
            }
            else{
                adjustRow(availableWidth, false);
            }
            imagesInCurrentRow = [];
        }
        /*
        $(this).velocity("fadeIn", {
            duration: Math.ceil(Math.random() * 3000)
        });
        */
        Log(Date.now(), "Image " + index + " loaded");
    });

    $('#container').append(image);

};

// Start
// ----------------------------------------------------------------------------------------------------------------------------------------

$(document).ready(function(){

    containerHeight = screenHeight;
    containerWidth = $("#container").width();
    availableWidth = $("#container").width() - 2 * marginWidth;

    console.log(screenHeight + '::' + screenWidth);
    var images = [];
    for (var i = 1; i <= totalNoOfImages; i++){
        images.push(i);
    }
    $.each(images, function(index, value){
        /*
        Images.push({
            imageIndex: value,
            placed: false
        });
        */
        
        var currentImage = new Image(images[index]);
        //currentImage.imageIndex = index;
        //currentImage.imagePath = "../static/images/im" + images[index]+ ".jpg";
        Images.push(currentImage);
    });
    
    var startIndex = 0;
    setImageInPlace(containerHeight, containerWidth, availableWidth, startIndex);
    
    
    
    // Handler: Done
    // -----------------------------------------
    
    $('#done').on('click', function(){
        recordEventsToFile();
    });

});