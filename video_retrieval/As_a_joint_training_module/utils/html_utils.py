import os

import dominate
from dominate.tags import a
from dominate.tags import attr
from dominate.tags import br
from dominate.tags import h3
from dominate.tags import img
from dominate.tags import meta
from dominate.tags import p
from dominate.tags import source
from dominate.tags import span
from dominate.tags import table
from dominate.tags import td
from dominate.tags import tr
from dominate.tags import video


class HTML:

  def __init__(self, web_dir, title, refresh=0):
    self.title = title
    self.web_dir = web_dir
    self.img_dir = os.path.join(self.web_dir, "images")
    if not os.path.exists(self.web_dir):
      os.makedirs(self.web_dir)
    if not os.path.exists(self.img_dir):
      os.makedirs(self.img_dir)

    self.doc = dominate.document(title=title)
    if refresh > 0:
      with self.doc.head:
        meta(http_equiv="refresh", content=str(refresh))

  def get_image_dir(self):
    """Return the directory that stores images."""
    return self.img_dir

  def add_header(self, text):
    with self.doc:
      h3(text)

  def add_videos(self, vids, txts, links, width=400, hidden_tag="hidden"):
    self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
    self.doc.add(self.t)
    colors = ["red", "blue", "gold", "salman"]
    with self.t:
      with tr():
        for vid, txt, link in zip(vids, txts, links):
          td_style = "word-wrap: break-word; width:{}px".format(width)
          with td(style=td_style, halign="center", valign="top"):
            with p():
              vid_path = str(vid)
              if vid_path == hidden_tag:
                p_style = "font-weight: bold; width:{}px;"
                p_style = p_style.format(width * 3)
                p("hidden video", style=p_style)
              else:
                with a(href=str(link)):
                  with video():
                    attr(controls="controls", width=width)
                    source(src=vid_path, type="video/mp4")
              br()
              rows = txt.split("<br>")
              for idx, row in enumerate(rows):
                color = colors[idx % len(colors)]
                bold_tag = "<b>"
                if not row.startswith(bold_tag):
                  s_style = "color:{};".format(color)
                else:
                  s_style = "color:black; font-weight: bold;"
                  row = row[len(bold_tag):]
                span(row, style=s_style)
                br()

  def add_images(self, ims, txts, links, width=400):
    self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
    self.doc.add(self.t)
    with self.t:
      with tr():
        for im, txt, link in zip(ims, txts, links):
          td_style = "word-wrap: break-word;"
          with td(style=td_style, halign="center", valign="top"):
            with p():
              with a(href=os.path.join("images", link)):
                img(
                    style="width:%dpx" % width,
                    src=os.path.join("images", im),
                )
              br()
              p(txt)

  def save(self):
    """Save the current content to the HMTL file."""
    html_file = "%s/index.html" % self.web_dir
    f = open(html_file, "wt")
    f.write(self.doc.render())
    f.close()


if __name__ == "__main__":  # we show an example usage here.
  html = HTML("web/", "test_html")
  html.add_header("hello world")

  imgs, texts, links1 = [], [], []
  for n in range(4):
    imgs.append("image_%d.png" % n)
    texts.append("text_%d" % n)
    links1.append("image_%d.png" % n)
  html.add_images(imgs, texts, links1)
  html.save()
