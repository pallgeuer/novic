window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {

	$(".navbar-burger").click(function() {
		$(".navbar-burger").toggleClass("is-active");
		$(".navbar-menu").toggleClass("is-active");
	});

	bulmaCarousel.attach('.carousel', {
		slidesToScroll: 1,
		slidesToShow: 3,
		loop: true,
		infinite: true,
		autoplay: false,
		autoplaySpeed: 3000,
	});

})
