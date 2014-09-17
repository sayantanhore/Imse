var sliderArray = [];
            
$(document).ready(function(){
    var container = $("<div id = 'container'></div>");
    $("body").append(container);

    container.offset({top: $(".navbar").css("height").replace("px", "")});
    container.css("margin-top", "0.5em");

    // Current

    container.append("<p id = 'header'>Current</p>");
    container.append("<div id = 'currentSetContainer' class = 'thumbnail'></div>");


    var currentImageSet = $("<div id = 'currentSet'></div>");

    for (var i = 0; i < 10; i ++){
        var imageContainer = $("<div></div>");
        var img = $("<img></img>");
        //img.attr("alt", "");
        var imPath = '../images/im' + parseInt(i + 1) + '.jpg';
        //console.log(imPath);
        img.attr("src", imPath);
        imageContainer.append(img);

        //Slider
        var sliderWrapper = $("<div></div>");
        sliderWrapper.addClass("slider-wrapper");
        var slider = new Slider();
        sliderWrapper.append(slider.createSlider());

        imageContainer.append(sliderWrapper);
        sliderArray.push(slider);
        sliderWrapper.hide();


        currentImageSet.append(imageContainer);
    }
    $("#currentSetContainer").append(currentImageSet);
    $("#currentSet").justifiedGallery();

    // Handlers for sliders

    $("body").on("mouseup", function(event){
        if (sliderPressed === true){
            //$(activeSlider.roller).trigger("mouseup");
            activeSlider.releaseSlider();
        }
    });

    $("body").on("mousemove", function(event){
        if (sliderPressed === true){
            activeSlider.dragSlider();
        }
    });

    console.log($(document).height());
    console.log($("#currentSetContainer").offset().top);
    console.log($("p").height());

    $("#currentSetContainer").height(parseFloat(($(document).height() - $("#currentSetContainer").offset().top) / 2) - 2 * $("p").height());
    $(".disp").width("1em");
    $("disp").css("font-size", "0.5em");

    // Future

    container.append("<p id = 'footer'>Future</p>");
    container.append("<div id = 'futureSetContainer' class = 'thumbnail'></div>");


    var futureImageSet = $("<div id = 'futureSet'></div>");

    for (var i = 10; i < 20; i ++){
        var imageContainer = $("<div></div>");
        var img = $("<img></img>");
        //img.attr("alt", "");
        var imPath = '../images/im' + parseInt(i + 1) + '.jpg';
        //console.log(imPath);
        img.attr("src", imPath);
        imageContainer.append(img);
        futureImageSet.append(imageContainer);
    }
    $("#futureSetContainer").append(futureImageSet);
    $("#futureSet").justifiedGallery();


    // Handlers

    $(".slider-wrapper").each(function(index, value){
        $(this).ready(function(){
            //console.log(sliderArray[index]);
            sliderArray[index].initSlider();
            sliderTop = $(this).parent().height() - $(this).height();
            //console.log("SliderTop :: " + $(this).parent().find("img").height());
            $(this).css("top", sliderTop);
        });
    });

    $(".slider-wrapper").parent().mouseenter(function(){

        $(this).find(".slider-wrapper").show();
    });

    $(".slider-wrapper").parent().mouseleave(function(){

        $(this).find(".slider-wrapper").hide();
    });
    $("#futureSetContainer").height(parseFloat(($(document).height() - $("#currentSetContainer").offset().top) / 2) - 2 * $("p").height());
});