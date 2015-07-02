"use script";

// Declare variables
// ----------------------------------------------------------------------------------------------------------------------------------------
var loc = window.location.pathname.substr(0, window.location.pathname.indexOf("/#"));
alert(window.location.pathname)
var screenHeight = $(window).height();
var screenWidth = parseInt($(window).width() * 100) / 100;

var imagesPerRow = 4;
var totalNoOfImages = 2 * 4;

var marginWidth = 2;
var availableWidth = 0;
var availableHeight = 0;

var Images = [];
var imageObjectOnFocus = null;

var __EVENT_ID__ = 0;
var __EVENTS__ = [];

// Create grid
// -----------------------------------------------

var createGrid = function(availableWidth, availableHeight){
    $("#container").css("width", screenWidth + "px");
    $("#container").css("height", screenHeight + "px");
    for (var i = 0; i < 8; i ++){
        var div = $("<div class = 'image-box'/>");
        $("#container").append(div);
        div.css("width", (parseFloat(availableWidth) / 4 - (0 * marginWidth)) + "px");
        div.css("height", (parseFloat(availableHeight) / 2 - (0 * marginWidth)) + "px");
        
    }
}

// Initiate FeedbackBox
// ----------------------------------------------------------------------------------------------------------------------------------------

var initiateFeedbackBox = function(feedbackBox, target){
    feedbackBox.css("left", "10px");
    feedbackBox.css("top", "10px")
    feedbackBox.css("background-color", "#DF3A01");
}

var loadImage = function(){
    //alert("Loading :: " + $(this).attr("src"));
    var width = $(this).width();
    var height = $(this).height();
    if (width >= height){
        $(this).width($(this).parent().width() - 10);
        var marginTop = parseFloat($(this).parent().height() - $(this).height()) / 2;
        console.log(marginTop)
        $(this).css("margin-left", 5 + "px");
        $(this).css("margin-top", marginTop + "px");
    }
    else{
        $(this).height($(this).parent().height() - 10);
        var marginLeft = parseFloat($(this).parent().width() - $(this).width()) / 2;
        $(this).css("margin-left", marginLeft + "px");
        $(this).css("margin-top", 5 + "px");
    }
    attachListeners($(this));
};

var attachFeedbackBox = function(event, changeAt){
    var target = $(event.target);
    /*
    if (imageObjectOnFocus !== null){
        imageObjectOnFocus.feedbackBox.remove();
        imageObjectOnFocus.feedbackBox = null;
        imageObjectOnFocus = null;
    }
    */
    imageObjectOnFocus = Images[changeAt];
    if (imageObjectOnFocus.feedbackBox === null){
        console.log("Ataching")
        var feedbackBox = $('<div class = "feedback-box"></div>');
        target.parent().append(feedbackBox);
        initiateFeedbackBox(feedbackBox, target);
        imageObjectOnFocus.feedbackBox = feedbackBox;
    }
    Log("IMAGE_MOUSEOVER", Date.now(), event.pageX, event.pageY, imageObjectOnFocus.imageIndex);
    imageObjectOnFocus.feedbackBox.html(imageObjectOnFocus.feedback);
    event.stopPropagation();
};

var feedbackBoxColourPicker = function(feedbackVal, imageObjectOnFocus){
    if (feedbackVal >= 0.0){
            imageObjectOnFocus.feedbackBox.css("background-color", "#DF3A01");
    }
    if (feedbackVal >= 3.0){
        imageObjectOnFocus.feedbackBox.css("background-color", "#DBA901");
    }
    if(feedbackVal >= 7.0){
        imageObjectOnFocus.feedbackBox.css("background-color", "#01DFA5");
    }
};

var removeFeedbackBox = function(event){
    imageObjectOnFocus.feedbackBox.remove();
    imageObjectOnFocus.feedbackBox = null;
    imageObjectOnFocus = null;
    event.stopPropagation();
};

var attachListeners = function(image){
    
    image.on('mouseover', function(event){
        var target = $(event.target);
        var changeAt = $("img").index(target);
        
        attachFeedbackBox(event, changeAt);
    });
    
    image.on('mouseout', function(event){
        Log("IMAGE_MOUSEOUT", Date.now(), event.pageX, event.pageY, imageObjectOnFocus.imageIndex);
        if (imageObjectOnFocus.feedback === 0.0){
            //alert("Removing")
            removeFeedbackBox(event);    
        }
        else{
            imageObjectOnFocus.feedbackBox.html(imageObjectOnFocus.feedback);
            feedbackBoxColourPicker(imageObjectOnFocus.feedback, imageObjectOnFocus);
        }
        
    });
    
    image.on('mousemove', function(event){
        
        var target = $(event.target);
        var mouse_x = event.pageX - target.offset().left;
        //target.trigger("mouseover");
        //alert(mouse_x + ':::' + target.width() + ':::' + parseFloat(target.width()) / 10);
        var feedbackVal = ((parseFloat(mouse_x) * 10.0 / (parseFloat(target.width())))).toFixed(1);
        feedbackBoxColourPicker(feedbackVal, imageObjectOnFocus);
        if (feedbackVal <= 0.0){
            imageObjectOnFocus.feedbackBox.html(0.0);
        }
        else if(feedbackVal >= 10.0){
            imageObjectOnFocus.feedbackBox.html(10.0);
        }
        else{
            imageObjectOnFocus.feedbackBox.html(feedbackVal);
        }
        
        //imageObjectOnFocus.feedback = parseFloat(imageObjectOnFocus.feedbackBox.html());
    });
    
    image.on('click', function(event){
        var target = $(event.target);
        imageObjectOnFocus.feedback = parseFloat(imageObjectOnFocus.feedbackBox.html());
        imageObjectOnFocus.feedbackBox.css("width", "4em");
        imageObjectOnFocus.feedbackBox.css("height", "4em");
        imageObjectOnFocus.feedbackBox.css("line-height", "4em");
        Log("IMAGE_CLICK", Date.now(), event.pageX, event.pageY, imageObjectOnFocus.imageIndex);
        Log("IMAGE_REMOVE", Date.now(), "N/A", "N/A", imageObjectOnFocus.imageIndex);
        /*
        var newImageIndex = -999;
        do{
            newImageIndex = Math.ceil(Math.random() * 100);
        }while(newImageIndex < totalNoOfImages);
        imageObjectOnFocus.changeImage(newImageIndex);
        */
        Log("IMAGE_LOAD", Date.now(), "N/A", "N/A", imageObjectOnFocus.imageIndex);
        event.stopPropagation();
    });
    
};


// Load initial images
// ---------------------------------------------------
var placeImages = function(images){
    var imageBoxes = $(".image-box");
    $.each(images, function(index, value){
        var currentImage = new Image(value, $(".image-box").eq(index));
        //currentImage.imageIndex = index;
        //currentImage.imagePath = "../static/images/im" + images[index]+ ".jpg";
        Images.push(currentImage);
        var imgPath = Images[index].imagePath;
        var img = $("<img/>").attr("src", imgPath);
        img.on("load", loadImage);
        currentImage.image = img;
        
        imageBoxes.eq(index).append(img);
        Log("IMAGE_LOAD", Date.now(), "N/A", "N/A", currentImage.imageIndex);
    });
}


// Start here
// -----------------------------------------------

var firstround = function(images){
    $.get("/firstround").done(function(data){
        data = data.results;
        
        console.log(data);
        for (var i = 0; i < data.length; i ++){
                images.push(parseInt(data[i]))
        }
        placeImages(images);
    });
}

var predict = function(images_shown, feedback){
    console.log("-------------------------------------")
    console.log(loc);
    console.log("-------------------------------------")
    alert(44)
    $.get("/search").done(function(data){
        data = data.results;
        
        console.log(data);
        for (var i = 0; i < data.length; i ++){
                images.push(parseInt(data[i]))
        }
        //Images = [];
        placeImages(images);
    });
}

$(document).ready(function(){
    availableWidth = screenWidth - (imagesPerRow * 2 + 4) * marginWidth;
    availableHeight = screenHeight - $('#title').height() - (2 * 2 + 8) * marginWidth;
    
    createGrid(availableWidth, availableHeight);
    
    var images = [];
    firstround(images);
    
    
    // Done
    // -----------------------------------------------

    $('#done').on('click', function(){
            Log("DONE_CLICKED", Date.now(), "N/A", "N/A", "N/A");
            //recordEventsToFile();
    });
    
    $('#next').on('click', function(){
            Log("NEXT_CLICKED", Date.now(), "N/A", "N/A", "N/A");
            var images_shown = [];
            var feedback = [];
            for (var i = 0; i < Images.length; i ++){
                images_shown.push(Images[i].imageIndex);
                feedback.push(Images[i].feedback);
            }
            predict(images_shown, feedback);
    });
    
});