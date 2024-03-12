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
    email: adamli@umich.edu
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
    url: https://arxiv.org/abs/2402.08857
  - icon: github
    icon-library: simpleicons
    text: Code
    url: https://github.com/roahmlab/sparrows

# End Front Matter
---

<!-- BEGIN DOCUMENT HERE -->

{% include sections/authors %}
{% include sections/links %}

---

# [Overview Videos](#overview-videos)

<!-- BEGIN OVERVIEW VIDEOS -->
<div class="fullwidth video-container" style="flex-wrap:nowrap; padding: 0 0.2em">
  <div class="video-item" style="min-width:0;">
    <video
      class="autoplay-on-load"
      preload="none"
      controls
      disablepictureinpicture
      playsinline
      muted
      loop
      style="display:block; width:100%; height:auto;"
      poster="assets/thumb/sparrows_single_arm_demo.jpg">
      <source src="assets/sparrows_single_arm_demo.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <p>SPARROWS performing single arm planning </p>
  </div>
  <div class="video-item" style="min-width:0;">
    <video
      class="autoplay-on-load"
      preload="none"
      controls
      disablepictureinpicture
      playsinline
      muted
      loop
      style="display:block; width:100%; height:auto;"
      poster="assets/thumb/sparrows_two_arm_demo.jpg">
      <source src="assets/sparrows_two_arm_demo.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <p>SPARROWS performing two arm planning </p>
  </div>
</div> <!-- END OVERVIEW VIDEOS -->

<!-- BEGIN ABSTRACT -->
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

</div> <!-- END ABSTRACT -->

<!-- BEGIN METHOD -->
<div markdown="1" class="justify">

# [Method](#method)

![link_construction](./assets/sfo_link_construction.png)
{: class="fullwidth"}

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

</div><!-- END METHOD -->

<!-- START RESULTS -->
<div markdown="1" class="content-block grey justify">

# [Simulation Results](#simulation-results)
## Random Scenarios
The following videos demonstrate the performance of SPARROWS to other methods in randomly generated hard schenarios.
In each of these, SPARROWS is able to acheive the desired goal configuration while the others don't.
ARMTD does stop in a safe configuration, but it gets stuck and fails make it to the goal.
On the other hand, MPOT and TRAJOPT both stop due to colliding with the environment.

<!-- START RANDOM VIDEOS -->
<div class="video-container">
  <div class="video-item">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      onclick="this.paused ? this.play() : this.pause();"
      poster="assets/thumb/combined_10_obstacles.jpg">
      <source src="assets/combined_10_obstacles.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <p>10 obstacles</p>
  </div>
  <div class="video-item">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      onclick="this.paused ? this.play() : this.pause();"
      poster="assets/thumb/combined_20_obstacles.jpg">
      <source src="assets/combined_20_obstacles.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <p>20 obstacles</p>
  </div>
  <div class="video-item">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      onclick="this.paused ? this.play() : this.pause();"
      poster="assets/thumb/combined_40_obstacles.jpg">
      <source src="assets/combined_40_obstacles.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <p>40 obstacles</p>
  </div>
</div><!-- END RANDOM VIDEOS -->


## Hard Scenarios
We also handcraft hard scenarios where the arm must go around large obstacles and maneuver through tight spaces.
SPARROWS' performance on a handful of these scenarios is demonstrated below.

<!-- START HARD VIDEOS -->
<div class="video-container">
  <div class="video-item tighter">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      onclick="this.paused ? this.play() : this.pause();"
      poster="assets/thumb/sparrows_hard_scenarios_2.jpg">
      <source src="assets/sparrows_hard_scenarios_2.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
  <div class="video-item tighter">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      onclick="this.paused ? this.play() : this.pause();"
      poster="assets/thumb/sparrows_hard_scenarios_3.jpg">
      <source src="assets/sparrows_hard_scenarios_3.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
  <div class="video-item tighter">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      onclick="this.paused ? this.play() : this.pause();"
      poster="assets/thumb/sparrows_hard_scenarios_8.jpg">
      <source src="assets/sparrows_hard_scenarios_8.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
  <div class="video-item tighter">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      onclick="this.paused ? this.play() : this.pause();"
      poster="assets/thumb/sparrows_hard_scenarios_4.jpg">
      <source src="assets/sparrows_hard_scenarios_4.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
  <div class="video-item tighter">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      onclick="this.paused ? this.play() : this.pause();"
      poster="assets/thumb/sparrows_hard_scenarios_11.jpg">
      <source src="assets/sparrows_hard_scenarios_11.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
</div><!-- END HARD VIDEOS -->
</div><!-- END RESULTS -->

<div markdown="1" class="justify">
  
# [Related Projects](#related-projects)
  
* [Autonomous Robust Manipulation via Optimization with Uncertainty-aware Reachability](https://roahmlab.github.io/armour/)
* [Reachability-based Trajectory Design with Neural Implicit Safety Constraints](https://roahmlab.github.io/RDF/)

<div markdown="1" class="content-block grey justify">
  
# [Citation](#citation)

This project was developed in [Robotics and Optimization for Analysis of Human Motion (ROAHM) Lab](http://www.roahmlab.com/) at University of Michigan - Ann Arbor.

```bibtex
@article{michaux2024sparrows,
  title={Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres},
  author={Jonathan Michaux and Adam Li and Qingyi Chen and Che Chen and Bohao Zhang and Ram Vasudevan},
  journal={ArXiv},
  year={2024},
  volume={abs/2402.08857},
  url={https://arxiv.org/abs/2402.08857}}
```
</div>


<!-- below are some special scripts -->
<script>
window.addEventListener("load", function() {
  // Get all video elements and auto pause/play them depending on how in frame or not they are
  let videos = document.querySelectorAll('.autoplay-in-frame');

  // Create an IntersectionObserver instance for each video
  videos.forEach(video => {
    const observer = new IntersectionObserver(entries => {
      const isVisible = entries[0].isIntersecting;
      if (isVisible && video.paused) {
        video.play();
      } else if (!isVisible && !video.paused) {
        video.pause();
      }
    }, { threshold: 0.25 });

    observer.observe(video);
  });

  // document.addEventListener("DOMContentLoaded", function() {
  videos = document.querySelectorAll('.autoplay-on-load');

  videos.forEach(video => {
    video.play();
  });
});
</script>

