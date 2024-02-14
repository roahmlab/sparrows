---
# Front matter. This is where you specify a lot of page variables.
layout: default
title:  "SPARROWS"
date:   2024-02-13 10:00:00 -0500
description: >- # Supports markdown
  Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres
show-description: true

# Add page-specifi mathjax functionality. Manage global setting in _config.yml
mathjax: false
# Automatically add permalinks to all headings
# https://github.com/allejo/jekyll-anchor-headings
autoanchor: false

# Preview image for social media cards
image:
  path: /assets/main_fig_compressed.jpg
  height: 600
  width: 800
  alt: SPARROWS Main Figure - 2-arm Planning

# Only the first author is supported by twitter metadata
authors:
  - name: Jonathan Michaux
    email: jmichaux@umich.edu
  - name: Adam Li
    <!-- url: https://buildingatom.io -->
    email: adamli@umich.edu
    <!-- mailto: adamli@umich.edu -->
  - name: Qingyi Chen
    email: chenqy@umich.edu
  - name: Che Chen
    email: cctom@umich.edu
  - name: Bohao Zhang
    email: jimzhang@umich.edu
  - name: Ram Vasudevan
    email: ramv@umich.edu

# If you just want a general footnote, you can do that too.
# See the sel_map and armour-dev examples.
author-footnotes:
  All authors affiliated with the department of Mechanical Engineering and Department of Robotics of the University of Michigan, Ann Arbor.

links:
  - icon: arxiv
    icon-library: simpleicons
    text: ArXiv
    url: https://arxiv.org/
  - icon: github
    icon-library: simpleicons
    text: Code
    url: https://github.com/roahmlab/spharmour

# End Front Matter
---
<!-- <head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Video Gallery</title>
<style>
    .video-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between; /* or space-around */
        margin: 0 -10px; /* Adjust according to your needs */
    }
    .video-item {
        width: calc(33.33% - 20px); /* Adjust the width according to your needs */
        margin: 10px;
    }
    .video-item video {
        width: 100%;
        height: auto;
    }
</style>
</head> -->

{% include sections/authors %}
{% include sections/links %}

---

# [Overview Videos](#overview-videos)
<div class="fullwidth video-container" style="display:flex;flex-wrap:nowrap;">
<div class="video-item">
<video preload="auto" disablepictureinpicture controls playsinline class="autoplay-in-frame" muted loop style="display: block; width:100%; height: auto;">
    <source src="assets/sparrows_single_arm_demo.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>
<p> SPARROWS performing single arm planning </p>
</div>
<div class="video-item">
<video preload="auto" disablepictureinpicture controls playsinline class="autoplay-in-frame" muted loop style="display: block; width:100%; height: auto;">
    <source src="assets/sparrows_two_arm_demo.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>
<p> SPARROWS performing two arm planning </p>
</div>
</div>

<div markdown="1" class="content-block justify grey">

# [Abstract](#abstract)
Generating safe motion plans in real-time is necessary for the wide-scale
deployment of robots in unstructured and human-centric environments. These
motion plans must be safe to ensure humans are not harmed and nearby objects are
not damaged. However, they must also be generated in realtime to ensure the
robot can quickly adapt to changes in the environment. Many trajectory
optimization methods introduce heuristics that trade-off safety and real-time
performance, which can lead to potentially unsafe plans. This paper addresses
this challenge by proposing Safe Planning for Articulated Robots Using
Reachability-based Obstacle Avoidance With Spheres (SPARROWS). SPARROWS is a
receding-horizon trajectory planner that utilizes the combination of a novel
reachable set representation and an exact signed distance function to generate
provably-safe motion plans. At runtime, SPARROWS uses parameterized trajectories
to compute reachable sets composed entirely of spheres that overapproximate the
swept volume of the robot’s motion. SPARROWS then performs trajectory
optimization to select a safe trajectory that is guaranteed to be
collision-free. We demonstrate that SPARROWS’ novel reachable set is
significantly less conservative than previous approaches.  We also demonstrate
that SPARROWS outperforms a variety of state-of-the-art methods in solving
challenging motion planning tasks in cluttered environments. Code will be
released upon acceptance of this manuscript.

</div>

<div markdown="1" class="justify">

# [Method](#method)

![link_construction](./assets/sfo_link_construction.png)

<!-- # Contributions -->
To address the limitations of existing approaches, this paper proposes Safe
Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With
Spheres (SPARROWS). The proposed method combines reachability analysis with
sphere-based collision primitives and an exact signed distance function to
enable real-time motion planning that is certifiably-safe, yet less conservative
than previous methods. This paper’s contributions are three-fold:
1. A novel reachable set representation composed of over- lapping spheres,
   called the Spherical Forward Occupancy (SFO), that overapproximates the
   robot’s reachable set and is differentiable;
2. An algorithm that computes the exact signed distance between a point and a
   three dimensional zonotope;
3. A demonstration that SPARROWS outperforms similar state-of-the-art methods on
   a set of challenging motion planning tasks

</div>

<div markdown="1" class="content-block grey justify">
# [Simulation Results](#simulation-results)

## Random Scenarios

The following videos demonstrate the performance of SPARROWS to other methods in randomly generated hard schenarios.
In each of these, SPARROWS is able to acheive the desired goal configuration while the others don't.
ARMTD does stop in a safe configuration, but it gets stuck and fails make it to the goal.
On the other hand, MPOT and TRAJOPT both stop due to colliding with the environment.

<div class="video-container">
    <div class="video-item">
        <video controls disablepictureinpicture playsinline muted class="autoplay-in-frame" loop>
            <source src="assets/combined_10_obstacles.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <p> 10 obstacles</p>
    </div>
    <div class="video-item">
        <video controls disablepictureinpicture playsinline muted class="autoplay-in-frame" loop>
            <source src="assets/combined_20_obstacles.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <p> 20 obstacles</p>
    </div>
    <div class="video-item">
        <video controls disablepictureinpicture playsinline muted class="autoplay-in-frame" loop>
            <source src="assets/combined_40_obstacles.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <p> 40 obstacles</p>
    </div>
    <!-- Repeat the above structure for more videos -->
</div>

## Hard Scenarios

We also handcraft hard scenarios where the arm must go around large obstacles and maneuver through tight spaces.
SPARROWS' performance on a handful of these scenarios is demonstrated below.

<div class="video-container">
    <div class="video-item tighter">
        <video controls disablepictureinpicture playsinline muted class="autoplay-in-frame" loop>
            <source src="assets/sparrows_hard_scenarios_2.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    <div class="video-item tighter">
        <video controls disablepictureinpicture playsinline muted class="autoplay-in-frame" loop>
            <source src="assets/sparrows_hard_scenarios_3.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    <div class="video-item tighter">
        <video controls disablepictureinpicture playsinline muted class="autoplay-in-frame" loop>
            <source src="assets/sparrows_hard_scenarios_4.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    <div class="video-item tighter">
        <video controls disablepictureinpicture playsinline muted class="autoplay-in-frame" loop>
            <source src="assets/sparrows_hard_scenarios_8.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    <div class="video-item tighter">
        <video controls disablepictureinpicture playsinline muted class="autoplay-in-frame" loop>
            <source src="assets/sparrows_hard_scenarios_11.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
</div>

</div>

<!-- <div markdown="1" class="content-block grey justify">
# [Citation](#citation)

*Insert whatever message*

```bibtex
@article{michaux2024sparrows,
  title={Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres},
  author={Jonathan Michaux and Adam Li and Qingyi Chen and Che Chen and Bohao Zhang and Ram Vasudevan},
  journal={ArXiv},
  year={2024},
  volume={},
}
```
</div> -->


<!-- below are some special scripts -->
<script>
  // Get all video elements
  const videos = document.querySelectorAll('.autoplay-in-frame');

  // Create an IntersectionObserver instance for each video
  videos.forEach(video => {
    const observer = new IntersectionObserver(entries => {
      const isVisible = entries[0].isIntersecting;
      if (isVisible && video.paused) {
        video.play();
      } else if (!isVisible && !video.paused) {
        video.pause();
      }
    }, { threshold: 0.7 });

    observer.observe(video);
  });
</script>

<!-- # [Content](#content) -->
<!-- <div markdown="1" class="content-block grey justify no-pre"> -->
<!-- some text -->

<!-- Try clicking this heading, this shows the manually defined header anchor, but if you do this, you should do it for all headings. -->
<!-- </div> -->

<!-- I made this look right by adding the `no-pre` class. -->
<!-- If you don't include `markdown="1"` it will fail to render any markdown inside. -->

<!-- You can also make fullwidth embeds (this doesn't actually link to any video) -->
<!-- <div class="fullwidth"> -->
<!-- <video controls="" style="background-color:black;width:100%;height:auto;aspect-ratio:16/9;"></video> -->
<!-- </div> -->

<!-- <div markdown="1" class="content-block grey justify"> -->
<!-- # Topic inside of the content block -->

<!-- Lorem ipsum dolor sit amet Consectetur adipiscing elit Integer molestie lorem at massa. -->

<!-- ![Alt Text](https://cdn.pixabay.com/photo/2019/09/05/01/11/mountainous-landscape-4452844_1280.jpg "Random Image") -->
<!-- </div> -->

<!-- # Topic outside of content block -->

<!-- ![Alt Text](https://cdn.pixabay.com/photo/2019/09/05/01/11/mountainous-landscape-4452844_1280.jpg "Random Image") -->

<!-- Lorem ipsum dolor sit amet Consectetur adipiscing elit Integer molestie lorem at massa. -->

<!-- ## This is how we can get the image at 100% -->

<!-- <div markdown="1" class="fullwidth"> -->
<!-- ![Alt Text](https://cdn.pixabay.com/photo/2019/09/05/01/11/mountainous-landscape-4452844_1280.jpg "Random Image") -->
<!-- </div> -->

<!-- ## And this is how we can get the image closer -->

<!-- <div markdown="1" class="no-pre"> -->
<!-- ![Alt Text](https://cdn.pixabay.com/photo/2019/09/05/01/11/mountainous-landscape-4452844_1280.jpg "Random Image") -->
<!-- </div> -->

<!-- Lorem ipsum dolor sit amet Consectetur adipiscing elit Integer molestie lorem at massa. -->

<!-- <div markdown="1" class="cabin"> -->
<!-- It's also possible to specify a new font for a specific section -->
<!-- </div> -->

<!-- <div markdown="1" class="jp"> -->
<!-- ## See? 1 -->
<!-- </div> -->

<!-- And you can also <span class="cabin">change it in the middle</span>, though that's a bit more problematic for other reasons. -->

<!-- To specify fonts, just use Google Fonts and update `_data/fonts.yml`. -->
<!-- Any fonts you add as extra fonts at the bottom become usable fonts in the body of the post. -->

<!-- There are also tools to grab icons from other repos. -->
<!-- Just use the following: -->
<!-- {% include util/icons icon='github' icon-library='simpleicons' -%} -->
<!-- , and you'll be able to add icons from any library you have enabled that is supported. -->

<!-- This uses the liquid template engine for importing. -->
<!-- If you include the - at the start of end of such a line, it say to discard all whitespace before or after. -->
<!-- In order to keep the comma there, we added the -. -->
<!-- This is what happens: -->
<!-- {% include util/icons icon='github' icon-library='simpleicons' %} -->
<!-- , when you don't have it (notice the space). -->

<!-- And if you have mathjax enabled in `_config.yml` or in the Front Matter as it is here, you can even add latex: -->

<!-- $$ -->
<!-- \begin{align*} -->
<!--   & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right) -->
<!--   = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\ -->
<!--   & (x_1, \ldots, x_n) \left( \begin{array}{ccc} -->
<!--       \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\ -->
<!--       \vdots & \ddots & \vdots \\ -->
<!--       \phi(e_n, e_1) & \cdots & \phi(e_n, e_n) -->
<!--     \end{array} \right) -->
<!--   \left( \begin{array}{c} -->
<!--       y_1 \\ -->
<!--       \vdots \\ -->
<!--       y_n -->
<!--     \end{array} \right) -->
<!-- \end{align*} -->
<!-- $$ -->

<!-- You can also treat a section of text as a block, and use kramdown's block attribution methods to change fonts. -->
<!-- You can see at the end of this section in the markdown that I do just that -->
<!-- {: class="cabin"} -->

<!-- <div markdown="1" class="content-block grey justify"> -->
<!-- # This is a really long heading block so I can see if justify breaks the heading, and make sure that headings don't get justify unless they are explicitly classed with justify like the following heading -->

<!-- # This is the following really long heading block so I can see if justify breaks the heading, and make sure that only this heading is justified because it has the explicit tag -->
<!-- {: class="justify"} -->
<!-- </div> -->

<!-- <div markdown="1" class="content-block grey justify"> -->
<!-- # Citation -->

<!-- *Insert whatever message* -->

<!-- ```bibtex -->
<!-- @article{nash51, -->
<!--   author  = "Nash, John", -->
<!--   title   = "Non-cooperative Games", -->
<!--   journal = "Annals of Mathematics", -->
<!--   year    = 1951, -->
<!--   volume  = "54", -->
<!--   number  = "2", -->
<!--   pages   = "286--295" -->
<!-- } -->
<!-- ``` -->
<!-- </div> -->
