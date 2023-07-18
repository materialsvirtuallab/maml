"""
Pyinvoke tasks.py file for automating releases and admin stuff.

Author: Shyue Ping Ong
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import re

import requests
from invoke import task
from monty.os import cd


@task
def make_doc(ctx):
    """
    Generate API documentation + run Sphinx.

    :param ctx:
    """

    # ctx.run("cp README.rst api-docs-source/index.rst")
    ctx.run("cp CHANGES.rst api-docs-source/changelog.rst")

    with cd("api-docs-source"):
        ctx.run("rm maml.*.rst", warn=True)
        ctx.run("sphinx-apidoc --separate -P -d 7 -o . -f ../maml")
        ctx.run("rm maml*.tests.*rst", warn=True)
        for f in glob.glob("maml*.rst"):
            newoutput = []
            with open(f) as fid:
                for line in fid:
                    if re.search(r"maml.*\._.*", line):
                        continue
                    else:
                        newoutput.append(line)

            with open(f, "w") as fid:
                fid.write("".join(newoutput))
        ctx.run("rm maml*._*.rst")

    ctx.run("rm -r docs", warn=True)

    ctx.run("sphinx-build -b html api-docs-source docs")

    with cd("docs"):
        ctx.run("rm -r .doctrees", warn=True)

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
    ctx.run('git commit -a -m "Update docs"')
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
def set_ver(ctx, version):
    lines = []
    with open("maml/__init__.py") as f:
        for l in f:
            if "__version__" in l:
                lines.append('__version__ = "%s"' % version)
            else:
                lines.append(l.rstrip())
    with open("maml/__init__.py", "w") as f:
        f.write("\n".join(lines) + "\n")

    lines = []
    with open("setup.py") as f:
        for l in f:
            lines.append(re.sub(r"version=([^,]+),", 'version="%s",' % version, l.rstrip()))
    with open("setup.py", "w") as f:
        f.write("\n".join(lines) + "\n")


@task
def release(ctx, version=datetime.datetime.now().strftime("%Y.%-m.%-d"), notest=True, nodoc=False):
    set_ver(ctx, version=version)
    ctx.run("rm -r dist build maml.egg-info", warn=True)
    if not notest:
        ctx.run("pytest maml")
    with open("CHANGES.rst") as f:
        contents = f.read()
    toks = re.split(r"\-+", contents)
    desc = toks[1].strip()
    toks = desc.split("\n")
    desc = "\n".join(toks[:-1]).strip()
    if not nodoc:
        make_doc(ctx)
        ctx.run("git add .")
        ctx.run('git commit -a -m "Update docs"')
        ctx.run("git push")

    payload = {
        "tag_name": "v" + version,
        "target_commitish": "master",
        "name": "v" + version,
        "body": desc,
        "draft": False,
        "prerelease": False,
    }
    response = requests.post(
        "https://api.github.com/repos/materialsvirtuallab/maml/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]},
    )
    print(response.text)
    ctx.run("rm -f dist/*.*", warn=True)
    ctx.run("python setup.py sdist bdist_wheel", warn=True)
    ctx.run("twine upload --skip-existing dist/*.whl", warn=True)
    ctx.run("twine upload --skip-existing dist/*.tar.gz", warn=True)
