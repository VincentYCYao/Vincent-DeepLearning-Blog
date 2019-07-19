---
layout: post
title:  "Review: Reducing Spatial Redundancy with Octave Convolution"
date:   2019-07-19 06:00:00 +0800
categories: CNN
---


In this blog, the paper [**Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution**](http://arxiv.org/abs/1904.05049) is reviewed. This paper was first published on arXiv in 2019.

The authors proposed **Octave Convolution (OctConv)**, which can replace vanilla convolution without changing the network structure. 

<br/><br/>






## Background and Inspiration

# Background

> Convolutional Neural Networks (CNNs) have achieved remarkable success in many computer vision tasks and their efficiency keeps increasing with recent efforts to reduce the inherent redundancy in dense model parameters and in the channel dimension of feature maps.

> Substantial redundancy also exists in the spatial dimension of the feature maps produced by CNNs


# Inspiration

> A natural image can be decomposed into a low spatial frequency component that describes the smoothly changing structure and a high spatial frequency component that describes the rapidly changing fine details 

> The output feature maps of a convolution layer can also be seen as a mixture of information at different frequencies.

**Octave Convolution** is proposed to factorize the mixed feature maps into low- and high- frequency features maps. The ratio **low-frequency features to high-frequency features** is defined by a hyper-parameter **`alpha`**. The proposed Octave Convolution can process and store the low- and high- feature maps in such a way that it can replace the conventional convolution operation without network structure adjustment.

![fig](/assets/190719_OctConv/img/fig1.jpg)

<br/><br/>












You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
