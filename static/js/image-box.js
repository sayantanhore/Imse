// ImageBox component

// Author Sayantan Hore
// Created on 19.08.2014

// --------------------------------

function ImageBox(){
    this.outerContainer = null;
    this.imageBoxArray = [];
    this.sliderArray = [];
}
ImageBox.prototype.createOuterContainer = function(){
    this.outerContainer = $("<div></div>");
    var topOuterContainer = $(".navbar").height();
    //console.log("Navbar height");
    //console.log(topOuterContainer);
    this.outerContainer.addClass("outer-image-container");
    $("body").append(this.outerContainer);
    this.outerContainer.css("margin-top", topOuterContainer);
    
}
ImageBox.prototype.createImageBox = function(){
    if (this.createOuterContainer === null){
        this.createOuterContainer();
    }
    var wrapper = $("<div></div>");
    wrapper.addClass("wrapper");
    var imageBox = $("<div></div>");
    imageBox.addClass("thumbnail");
    var closeButton = $("<div>Accepted</div>");
    closeButton.addClass("btn-close");
    
    imageBox.append(closeButton);
    //closeButton.hide();
    
    var sliderWrapper = $("<div></div>");
    sliderWrapper.addClass("slider-wrapper");
    var slider = new Slider();
    sliderWrapper.append(slider.createSlider());
    
    imageBox.append(sliderWrapper);
    sliderWrapper.hide();
    this.sliderArray.push(slider);
    wrapper.append(imageBox);
    this.imageBoxArray.push(wrapper);
    return wrapper;
}

ImageBox.prototype.loadImage = function(imPath, imageBox){
    var img = $("<img></img>")
    img.attr("src", imPath);
    imageBox.find(".thumbnail").append(img);

    img.load(function(){
        //console.log($(this).height());
        imgWidth = $(this).width();
        $(this).parent().height(rowHeight);
        $(this).height(rowHeight);
        //console.log($(this).width());
        $(this).parent().width($(this).width());
        var closeBtn = $(this).parent().find(".btn-close");
        
        //var closeBtnLeft = $(this).parent().width() - closeBtn.width();
        //console.log(closeBtnLeft);
        //closeBtn.css("left", closeBtnLeft);
        sliderTop = $(this).closest(".thumbnail").height() - $(this).siblings(".slider-wrapper").height();
        
        $(this).siblings(".slider-wrapper").css("top", sliderTop);
        
        //closeBtn.show();
    });
}