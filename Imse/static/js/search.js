// Globals

var username;
var firstIteration = true;
var loc = window.location.pathname.substr(0, window.location.pathname.indexOf("/start"));
var images_loaded = 0;
var no_of_shown_images = 12;
var feedback = [];
var sliders = [];
var images_accepted = [];
var current_image_accepted = false;

var image_container_width;


// Global Method: Convert RGB to HEX

function rgb2hex(rgb) {
	rgb = rgb.match(/^rgb\((\d+),\s*(\d+),\s*(\d+)\)$/);
	function hex(x) {
	    return ("0" + parseInt(x).toString(16)).slice(-2);
	}
	return "#" + hex(rgb[1]) + hex(rgb[2]) + hex(rgb[3]);
}




// Global - Method: Get slider value

function getFeedback(val){
	console.log("Get Feedback Called :: ");
	feedback = [];
	var slider_input = $("#id_image_container .panel-body .row .col-md-3 input[type = text]")
	slider_input.each(function(){
    		feedback.push($(this).data("slider").getValue() / 10);
    });
    
    console.log("Feedback :: " + feedback);
    var no_of_images_accepted = images_accepted.length;

    if(no_of_images_accepted === 0){
    	addInset(no_of_images_accepted);
    }
    else{
    	if(images_accepted[no_of_images_accepted - 1].accepted === true){
    		addInset(no_of_images_accepted);
    	}
    }
    console.log("Feedback :: " + feedback);
    predict();
}



// Global Method: Show-Hide Mask

function pageMask(mask_status, mask_elem){
	if(mask_status === true){
		mask_elem.css("display", "block");
	}
	else{
		mask_elem.css("display", "none");
	}
}




// Global Method: Create Imagebox

function addInset(no_of_images_accepted){
	var imageBox = createImageBox(true).imageBox;
	images_accepted.push(function(){

		return {
			"imageBox": imageBox,
			"imageFileNo": "",
			"accepted": false,
		}

	});

	//$("#id_image_container .panel-body .row .col-md-3:eq(" + no_of_images_accepted + ")").append(imageBox);
	//$(".inset").css("left", $("#id_image_container .panel-body .row .col-md-3:eq(" + no_of_images_accepted + ") a").css("left"));
	var no_of_present_images = $("#id_image_container .panel-body .row .col-md-3").length;
	if(no_of_present_images % 4 === 0){
		$("#id_image_container .panel-body .row:last-child").css("padding-bottom", "0em");
		var row = $("<div class = 'row'></div>");
		row.append(imageBox);
		row.css("padding-bottom", "1.5em");
		$("#id_image_container .panel-body").append(row);
	}
	else{
		$("#id_image_container .panel-body .row:last-child").append(imageBox);
	}
	setImageContainerStyle($("#id_image_container .panel-body .row:last-child .col-md-3:last-child"));
	//$("#id_image_container .panel-body .row:last-child").css("padding-bottom", "-1.5em");
	console.log("Inset Added :: " + no_of_present_images);
}




// Global Method: Set Image Container's Styles

function setImageContainerStyle(elem){
	elem.find(".thumbnail").css("height", image_container_width).addClass("thumbnail-modifier");
	elem.find("input[type = text]").each(function(){
		var slider_width = $(this).parent().width() * 80 / 100;
		$(this).width(slider_width);
		var padding_left = ($(this).parent().width() - $(this).width()) / 2;
		$(this).parent().css("padding-left", padding_left + "px");
		$(this).parent().css("top", 9 * $(this).parent().width() / 10 + "px");
		$(this).parent().css("display", "none");
	});
}




// Global Method: Configure Images after Loading

function loadImage(){
	var img_width = $(this).width();
	var img_height = $(this).height();
	var width_to_set = image_container_width.replace("px", "");

	if(img_width >= img_height){
		$(this).width(width_to_set * 97 / 100);
	}
	else if(img_width < img_height){
		$(this).height(width_to_set * 97 / 100);
	}

	padding = (width_to_set - img_height) / 2;
	$(this).css("padding-top", padding + "px");

	$(this).parent().parent().mouseover(function(event){
		event.stopPropagation();
		$(this).find(".slider-header").css("display", "block");
	});

	$(this).parent().parent().mouseout(function(event){
		event.stopPropagation();
		$(this).find(".slider-header").css("display", "none");
	});

	images_loaded ++;

	if(images_loaded === no_of_shown_images){
		pageMask(false, $("#id_mask"));
		images_loaded = 0;
	}
};




// Global Method: Ajax Call for Image Prediction

function predict(){
    	var input_data = {
    		"username": username,
    		"algorithm": "GP-SOM",
    		"imagesnumiteration": no_of_shown_images,
    		"category": "None",
    		"feedback": JSON.stringify(feedback),
    		"accepted": current_image_accepted
    	};

    	if(current_image_accepted === true){
    		current_image_accepted = false;
    	};

    	if(firstIteration == true){
    		url = loc + "/search/start/"
    		firstIteration = false;
    	}
    	else{
    		url = loc + "/search/"
    	}
    	//pageMask(true, $("#id_mask"));
    	$.get(url, input_data).done(function(data){
            console.log("Incoming image")
            console.log(typeof eval(data))
            console.log(eval(data))
    		console.log(parseInt(eval(data)[0]))
    		var new_image = $(document).find(".glyphicon-ok").parent().parent().find("img");
            console.log(parseInt(data[0]) + 1)
			new_image.attr("src", loc + "/media/im" + (parseInt(eval(data)[0]) + 1) + ".jpg");
			new_image.on("load", loadImage);
		}).fail(function(){
			pageMask(false, $("#id_mask"));
			console.log(arguments);
		});

		// Scroll to see last image
		
		$("#id_image_container .panel-body").animate({
            scrollTop: $("#id_image_container .panel-body .row:last-child").offset().top
        }, 500);
		
    //});
	};


//******************************************************************************************************************************************************************************************************************************************************************************


$(document).ready(function(){

	// Page Mask

    $("#id_mask p").css("padding-top", $(document).height() / 2 + "px");

    // Blink Mask

    (function blink(){
	    $("#id_mask p").animate({
	        opacity: "0"
	    }, function(){
	        $(this).animate({
	            opacity: "1"
	        }, blink);
	    });
	})();




	// Adjusting height both Left & Right sides
	//**************************************************************************************************************************************************************************************************************************************************************************

	adjustPanelHeight = function(){
		$("#id_left_sidebar > .panel-body").css("height", ($(document).height() - $("#id_navbar").height()) + "px");
		$("#id_image_container > .panel-body").css("height", ($(document).height() - $("#id_navbar").height()) + "px");
	}

	//adjustPanelHeight();

	//**************************************************************************************************************************************************************************************************************************************************************************




	// Adjusting color-palette width & height to make it a sqare box
	//**************************************************************************************************************************************************************************************************************************************************************************

	setTableSqareBox = function(){
		$(".table-condensed").css("max-height", $(".table-condensed").css("width"));
	}
	
	//setTableSqareBox();

    //**************************************************************************************************************************************************************************************************************************************************************************




	// Adjusting color palette
	//**************************************************************************************************************************************************************************************************************************************************************************

	// Declare color placeholder

    var color_place_holder = ["", "", "", "", "", 0];

	adjustColorPalette = function(){
		// Load colors onclick palette

	    $("#id_table_color_palette td").each(function(){
	            $(this).click(function(){
	                bc = $(this).css("background-color");
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
	        box_width = space_available / 16.0;
			$(this).css("width", box_width)
			$(this).css("height", box_width)
			if(index != 4){
				$(this).css("margin-right", (space_available - box_width * 5) / 5  + "px");
			}
	        
			
		});

		
		
	    $("#id_table_color_palette td div").each(function(){
	        $(this).css("height", $(this).width() + "px");
	    });
		
	    $(".container-fluid").css("opacity", "0.0");
	    $('#id_login').modal('show');
	}
	
	//adjustColorPalette();

    //**************************************************************************************************************************************************************************************************************************************************************************





	// Adding image containers
	//**************************************************************************************************************************************************************************************************************************************************************************

	addInitialEmptyImageContainers = function(){
		var no_of_columns = parseInt(no_of_shown_images / 4) + parseInt(no_of_shown_images % 4);
		console.log(no_of_columns);
	    for (i = 0; i < no_of_columns ; i++){
	    	var row = $("<div class = 'row'></div>");
	    	for (j = 0; j < 4; j++){
	    		
	    		if(4 * (i) + (j) < no_of_shown_images){
	    			row.append(createImageBox(false).imageBox);
	    		}
	    		
	    		feedback.push(0);
	    	}
	    	$("#id_image_container .panel-body").append(row);
	    }
	    
	    // Set padding-bottom for last row to make it clearly visible

	    $("#id_image_container .panel-body .row:last-child").css("padding-bottom", "1.5em");

	    // Initialize image container's width

	    image_container_width = $("#id_image_container .panel-body .row .col-md-3 .thumbnail").css("width");


	    // Restyle images

	    setImageContainerStyle($("#id_image_container .panel-body .row .col-md-3"));

	}
	
	//addInitialEmptyImageContainers();

    //**************************************************************************************************************************************************************************************************************************************************************************

    
    // Setup
    //**************************************************************************************************************************************************************************************************************************************************************************
    
    init = function(){
    	adjustPanelHeight();
    	setTableSqareBox();
    	adjustColorPalette();
    	addInitialEmptyImageContainers();
    }

    init();
    //**************************************************************************************************************************************************************************************************************************************************************************




    //**************************************************************************************************************************************************************************************************************************************************************************
    // Event Handlers
	//**************************************************************************************************************************************************************************************************************************************************************************


	

    // Accept an image

    $(document).on("click", ".glyphicon-ok", function(){
    	// Catch the parent

    	var parent_image_container = $(this).parents(".col-md-3");
    	
    	// Put the image on accepted list

    	images_accepted[images_accepted.length - 1].accepted = true;
    	//$(this).siblings(".glyphicon-remove").remove();

    	// Remove header with tick icon

    	$(this).parent().remove();
    	
    	// Attch a slider to the accepted image
    	
    	parent_image_container.append(attachSlider());

    	// Restyle accepted image

    	setImageContainerStyle(parent_image_container);
    	
    	// Push the new slider into sliders list
    	sliders.push(parent_image_container.find("img").parent().siblings().find("input[type = text]").slider("setValue", 0)[0]);
    	
    	// Set accepted status true
    	current_image_accepted = true;

    	/*
    	sliders.each(function(){
    		$(this).data("slider").setValue(0);
    	});
		*/

		
		/*
    	if(images_accepted.length === no_of_shown_images){
    		$(".navbar-right > li:first-child a").css("display", "block")
    	}
    	*/
    });


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
		pageMask(true, $("#id_mask"));
		console.log("Testing URL:: " + loc)
    	$.get(loc + "/firstround", {colors: JSON.stringify(colors), no_of_images: no_of_shown_images}).done(function(data){
    		
			console.log("Success :: " + data);
			data = data.replace("[", "");
			data = data.replace("]", "")
			data = data.split(", ")
			var images = $("#id_image_container img")
			
			$("#id_image_container img").each(function(index){
				image_pos = parseInt(data[index]) + 1;
				// Loading images
				$(this).attr("src", loc + "/media/im" + image_pos + ".jpg");
				$(this).on("load", loadImage);

			});
			// Generate Sliders
			sliders = $("#id_image_container img").parent().siblings().find("input[type = text]").slider("setValue", 0);
			console.log(sliders.length)
			
		}).fail(function(){
			pageMask(false, $("#id_mask"));
			console.log(arguments);
		});
    });



	
    // Loading predicted images

    $(".navbar-right > li:first-child a").click(function(event){
    	alert("Do you want to accept? ");
    	var new_image_sources = [];

    	// Collect image src's for temporary images
    	$(".inset img").each(function(){
    		console.log("Now :: ")
    		new_image_sources.push($(this).attr("src"));
    	});

    	//Replace old images with newer ones
    	$("#id_image_container a img").each(function(index){

    			// Replace images
				$(this).attr("src", new_image_sources[index]);
				$(this).on("load", loadImage);

				// Clear stored temporary images
				images_accepted = [];

				// Remove insets
				$(".inset").remove();
		});

    	sliders.each(function(){
    		$(this).data("slider").setValue(0);
    	});

    	$(this).css("display", "none");

    });


    //$(".navbar-right a").click(function(event){
 
});
