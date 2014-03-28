$(document).ready(function(){

	// Globals

	var username;
	var firstIteration = true;


	// Adding images to the image container

    for (i = 0; i < 3 ; i++){
    	var row = $("<div class = 'row'></div>");
    	for (j = 0; j < 4; j++){
    		var image = $("<img src='' alt='...'>");
    		var link = $("<a href='#' class='thumbnail'>");

    		var input_slider = $("<input type='text' class='' value='' data-slider-min='0' data-slider-max='10' data-slider-step='1' data-slider-value='5' data-slider-orientation='horizontal' style='margin-top: 50px' data-slider-handle='round'>");
    		var slider_container = $("<div style='position: relative;' class = 'slider-header'>");
    		slider_container.append(input_slider);

    		var column = $("<div class = 'col-md-3'>");
    		link.append(image);
    		column.append(slider_container);
    		column.append(link);
    		row.append(column);
    	}
    	$("#id_image_container .panel-body").append(row);
    }

	// Convert RGB to HEX	
	function rgb2hex(rgb) {
		rgb = rgb.match(/^rgb\((\d+),\s*(\d+),\s*(\d+)\)$/);
		function hex(x) {
		    return ("0" + parseInt(x).toString(16)).slice(-2);
		}
		return "#" + hex(rgb[1]) + hex(rgb[2]) + hex(rgb[3]);
	}

    // Declare color placeholder
    var color_place_holder = ["", "", "", "", "", 0];
    
	$("#id_left_sidebar > .panel-body").css("height", ($(document).height() - $("#id_navbar").height()) + "px");
	$("#id_image_container > .panel-body").css("height", ($(document).height() - $("#id_navbar").height()) + "px");
	var color_palette_width = $(".table-condensed").css("width");
	console.log(color_palette_width)
	$(".table-condensed").css("max-height", color_palette_width)
    //$(".table-responsive").css("height", color_palette_width)
    
	var image_container_width = $("#id_image_container .panel-body .row .col-md-3 .thumbnail").css("width");
	$("#id_image_container .panel-body .row .col-md-3 .thumbnail").css("height", image_container_width);
	$("#id_image_container .panel-body .row .col-md-3 .thumbnail").addClass("thumbnail-modifier");
	

	// Configure images after loading

	var loadImage = function(){
		var img_width = $(this).width();

		var img_height = $(this).height();

		var width_to_set = image_container_width.replace("px", "");
		console.log("image width :: " + img_width);
		console.log("image height :: " + img_height)

		if(img_width >= img_height){
			console.log(width_to_set);
			$(this).width(width_to_set * 97 / 100);
			//$(this).css("width", image_container_width);
			
			
		}
		else if(img_width < img_height){
			$(this).height(width_to_set * 97 / 100);
			//$(this).css("height", image_container_width);
		}
		padding = (width_to_set - img_height) / 2;
		$(this).css("padding-top", padding + "px");
		//$(this).next().css("width", image_container_width);
		
		$(this).parent().parent().mouseover(function(event){
			event.stopPropagation();
			$(this).find(".slider-header").css("display", "block");
		});
		
		$(this).parent().parent().mouseout(function(event){
			event.stopPropagation();
			$(this).find(".slider-header").css("display", "none");
		});
		if($(this).parent().siblings().find("input[type = text]").length !== 0){
			$(this).parent().siblings().find("input[type = text]").slider("setValue", 0);
			console.log($(this).parent().siblings().find("input[type = text]").data("slider").getValue() / 10);
		}
		//console.log($(this).attr("src").replace("/media/", ""))
	};
    
	var img = $("#id_image_container img");

	img.on("load", loadImage);

	// Set padding to image containing rows

	$(".col-md-3 .panel-body").height($(".col-md-9 .panel-body").height());
    // Load colors onclick palette

    $("#id_table_color_palette td").each(function(){
            $(this).click(function(){
                bc = $(this).css("background-color");
                console.log(bc)
                $("#id_colors_picked .thumbnail").each(function(index, element){
                    if(color_place_holder[5] === 5){
                        alert("More than 5 choices are not allowed");
                        return false;
                    } 
                    else if(color_place_holder[index] === ""){
                        $(this).tooltip("enable");
                        color_place_holder[index] = bc;
                        $(this).css("background-color", bc);
                        color_place_holder[5] += 1;
                        return false;
                    }
                });
            });
    });
    
    $("#id_colors_picked .thumbnail").each(function(index, element){
        $(this).click(function(){
            if(color_place_holder[index] !== ""){
                $(this).tooltip("disable");
                $(this).css("background-color", "#424242");
                color_place_holder[index] = "";
                color_place_holder[5] -= 1;
                return false;
            }
            
        });
    });

	$("#id_colors_picked .thumbnail").each(function(index, element){
        $(this).tooltip("disable");
		var space_available = $(this).parent().css("width").replace("px", "");
		console.log("Space :: " + space_available);
        box_width = space_available / 16.0;
        console.log("Box width :: " + box_width);
		$(this).css("width", box_width)
		$(this).css("height", box_width)
		if(index != 4){
			$(this).css("margin-right", (space_available - box_width * 5) / 5  + "px");
            //$(this).css("margin-right", 20 + "px");
		}
        
		
	});

	$("#id_image_container .panel-body .row .col-md-3 input[type = text]").each(function(){
		var slider_width = $(this).parent().width() * 80 / 100;
		
		$(this).width(slider_width);
		var padding_left = ($(this).parent().width() - $(this).width()) / 2;
		//var padding_top = ($(this).parent().height() - $(this).height()) / 2;
		console.log("Slider-width :: " + padding_left)
		//$(this).css("left", slider_width + "px");
		console.log("ID :: " + $(this).parent().attr("id"))
		$(this).parent().css("padding-left", padding_left + "px");
		$(this).parent().css("top", 9 * $(this).parent().width() / 10 + "px");
		$(this).parent().css("display", "none");
	});
	/*
	$('.slider').slider({
			//'setValue': 0,
          	formater: function(value) {
          		return 'Current value: '+value;
          	}
	});
	*/
    $("#id_table_color_palette td div").each(function(){
        $(this).css("height", $(this).width() + "px");
    });
	//$('id_text').val($('.slider').slider())
	//$("#id_color_container").load("color_palette.html")
    
    // Call Log In screen
    //var login_height = $(".modal-dialog").height();
    //console.log("LOGIN Height :: " + login_height)
    //$("#id_login").css("margin-top", ($(document).height()) / 2 + "px");
    $(".container-fluid").css("opacity", "0.0");
    $('#id_login').modal('show');

    // Checking for a valid username
    $(".btn-modal").click(function(){
    	if($("#id_username").val() === ""){
    		$("#id_username").parent().parent().addClass("has-error");
    	}
    	else{
    		username = $("#id_username").val();
	    	brand_text = $(".navbar-brand").text();
	    	brand_text = brand_text + " [Welcome " +  username+ "]"
	    	$(".navbar-brand").text(brand_text);
	    	$(".container-fluid").css("opacity", "1.0");
	    	$('#id_login').modal('hide');
    	}
    });

    // Removing error class from text input
    $("#id_username").on("keypress", function(event){
    	$(this).parent().parent().removeClass("has-error");
    });

    // Loading initial images
    
    $("#id_proceed").click(function(event){
    	colors = [];
    	for (i = 0; i < color_place_holder[5]; i++){
			if(color_place_holder[i] !== ""){
				colors.push(rgb2hex(color_place_holder[i]));
			}
		}

    	$.get("/firstround/", {colors: JSON.stringify(colors), no_of_images: 12}).done(function(data){

			console.log("Success :: " + data);
			data = data.replace("[", "");
			data = data.replace("]", "")
			data = data.split(", ")
			var images = $("#id_image_container img")
			console.log(images.length)
			$("#id_image_container img").each(function(index){
				image_pos = parseInt(data[index]) + 1;
				$(this).attr("src", "/media/im" + image_pos + ".jpg")
				$(this).on("load", loadImage);
			});
			
		}).fail(function(){
			console.log("Failure");
		});
    });
	
    // Loading predicted images
    
    $(".navbar-right a").click(function(event){
    	var feedback = [];
    	$("#id_image_container .panel-body .row .col-md-3 input[type = text]").each(function(){
    		feedback.push($(this).data("slider").getValue() / 10);
    		//feedback.push(10);
    	});
    	console.log(feedback)
    	var input_data = {
    		"username": username,
    		"algorithm": "GP-SOM",
    		"imagesnumiteration": "12",
    		"category": "None",
    		"feedback": JSON.stringify(feedback)
    	};
    	
    	if(firstIteration == true){
    		url = "/search/start/"
    		firstIteration = false;
    	}
    	else{
    		url = "/search/"
    	}

    	$.get(url, input_data).done(function(data){

			console.log("Success :: " + data);
			data = data.replace("[", "");
			data = data.replace("]", "")
			data = data.split(", ")
			var images = $("#id_image_container img")
			console.log(images.length)
			$("#id_image_container img").each(function(index){
				image_pos = parseInt(data[index]) + 1;
				$(this).attr("src", "/media/im" + image_pos + ".jpg")
				$(this).on("load", loadImage);
			});
			
		}).fail(function(){
			console.log("Failure");
		});
    });
	
    
});