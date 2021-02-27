"""
Pyinvoke tasks.py file for automating releases and admin stuff.

Author: Shyue Ping Ong
"""

from invoke import task
import glob
import os
import json
import webbrowser
import requests
import re
import subprocess
import datetime

from monty.os import cd


NEW_VER = datetime.datetime.today().strftime("%Y.%-m.%-d")


@task
def make_doc(ctx):
    """
    Generate API documentation + run Sphinx.

    :param ctx:
    """

    ctx.run("cp README.rst api-docs-source/index.rst")
    ctx.run("cp CHANGES.rst api-docs-source/changelog.rst")

    with cd("api-docs-source"):
        ctx.run("rm maml.*.rst", warn=True)
        ctx.run("sphinx-apidoc --separate -P -d 7 -o . -f ../maml")
        ctx.run("rm maml*.tests.*rst", warn=True)
        for f in glob.glob("maml*.rst"):
            newoutput = []
            with open(f, 'r') as fid:
                for line in fid:
                    if re.search("maml.*\._.*", line):
                        continue
                    else:
                        newoutput.append(line)

            with open(f, 'w') as fid:
                fid.write("".join(newoutput))
        ctx.run("rm maml*._*.rst")

    ctx.run("rm -r docs", warn=True)

    ctx.run("sphinx-build -b html api-docs-source docs")

    # ctx.run("cp _static/* ../docs/html/_static", warn=True)

    with cd("docs"):
        ctx.run("rm -r .doctrees", warn=True)

        # This makes sure maml.org works to redirect to the Github page
        ctx.run("echo \"maml.ai\" > CNAME")
        # Avoid the use of jekyll so that _dir works as intended.
        ctx.run("touch .nojekyll")

@task
def update_doc(ctx):
    """
    Update the web documentation.

    :param ctx:
    """
    ctx.run("cp README.rst docs/")
    ctx.run("cp api-docs-source/conf.py docs/conf.py")
    make_doc(ctx)
    ctx.run("git add .")
    ctx.run("git commit -a -m \"Update docs\"")
    ctx.run("git push")


@task
def publish(ctx):
    """
    Upload release to Pypi using twine.

    :param ctx:
    """
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python setup.py sdist bdist_wheel")
    ctx.run("twine upload dist/*")


@task
def set_ver(ctx):
    lines = []
    with open("maml/__init__.py", "rt") as f:
        for l in f:
            if "__version__" in l:
                lines.append('__version__ = "%s"' % NEW_VER)
            else:
                lines.append(l.rstrip())
    with open("maml/__init__.py", "wt") as f:
        f.write("\n".join(lines))

    lines = []
    with open("setup.py", "rt") as f:
        for l in f:
            lines.append(re.sub(r'version=([^,]+),', 'version="%s",' % NEW_VER,
                                l.rstrip()))
    with open("setup.py", "wt") as f:
        f.write("\n".join(lines))


@task
def release_github(ctx):
    with open("CHANGES.rst") as f:
        contents = f.read()
    toks = re.split(r"\#+", contents)
    desc = toks[1].strip()
    payload = {
        "tag_name": "v" + NEW_VER,
        "target_commitish": "master",
        "name": "v" + NEW_VER,
        "body": desc,
        "draft": False,
        "prerelease": False
    }
    response = requests.post(
        "https://api.github.com/repos/materialsvirtuallab/maml/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]})
    print(response.text)


@task
def release(ctx, notest=False):
    ctx.run("rm -r dist build maml.egg-info", warn=True)
    if not notest:
        ctx.run("pytest maml")
    publish(ctx)
    release_github(ctx)
