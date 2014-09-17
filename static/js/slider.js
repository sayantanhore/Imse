// A simple draggable slider control created using jQuery
// Author: Sayantan Hore
// 18.08.2014

var activeSlider = null;
var sliderPressed = false;

function Slider(){
    this.xOffset = 0.0;
    this.rollerRadius = 0.0;
    this.slider;
    this.display;
    this.axel;
    this.roller;
}

// Create the slider
Slider.prototype.createSlider = function(){
    this.slider = $("<div></div>");
    this.slider.addClass("slider-container");

    this.display = $("<div></div>");
    this.display.addClass("disp");
    this.slider.append(this.display);

    this.axel = $("<div></div>");
    this.axel.addClass("axel");
    this.slider.append(this.axel);

    this.roller = $("<div></div>");
    this.roller.addClass("roller");
    this.axel.append(this.roller);
    //console.log(this.axel.height());
    return this.slider;
}

// Initialize slider
Slider.prototype.initSlider = function(){
    
    var rollerTopPosition = parseFloat((this.roller.height() - this.axel.height()) / 2);
    //console.log(rollerTopPosition);
    this.rollerRadius = parseFloat(this.roller.width() / 2);
    this.roller.css("top", -rollerTopPosition);
    this.roller.css("left", -this.rollerRadius);
    this.display.html("0.00");
    
    var sliderObj = this;
    
    this.roller.on("mousedown", function(event){
        sliderObj.xOffset = event.pageX - $(this).offset().left;
        sliderPressed = true;
        activeSlider = sliderObj;
        $(this).css("opacity", "0.6");
        event.stopPropagation();
    });
    this.roller.on("mouseup", function(event){
        sliderObj.releaseSlider();
    });
}

// Drag slider
Slider.prototype.dragSlider = function(){
    moveToX = event.pageX - this.axel.offset().left - this.xOffset;
    if ((moveToX + this.rollerRadius) >= 0 && (moveToX + this.rollerRadius) <= (this.axel.width())){
        this.roller.css("left", (moveToX));
        this.display.html(parseFloat((moveToX + this.rollerRadius) / this.axel.width()).toFixed(2));
    }
}

// Release slider
Slider.prototype.releaseSlider = function(){
    sliderPressed = false;
    this.roller.css("opacity", "1.0");
    activeSlider = null;
    event.stopPropagation();
}


