---
layout: default
---

<div>

  {{ content }}

  <h1>Deep Learning Research</h1>
  <ul class="post-list">
    {% for post in site.posts %}
      {% if post.path contains 'DeepLearning_posts'%}
        <li>
          {% assign date_format = site.cayman-blog.date_format | default: "%b %-d, %Y" %}
          <span class="post-meta">{{ post.date | date: date_format }}</span>
          <h2>
            <a class="post-link" href="{{ post.url | absolute_url }}" title="{{ post.title }}">{{ post.title | escape }}</a>
          </h2>
          <!-- Uncomment the below line to add preview of blog -->
          {{ post.excerpt | markdownify | truncatewords: 25 }}
        </li>
      {% endif %}
    {% endfor %}
  </ul>

</div>
