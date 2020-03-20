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
    # with open("CHANGES.rst") as f:
    #     contents = f.read()
    #
    # toks = re.split(r"\-{3,}", contents)
    # n = len(toks[0].split()[-1])
    # changes = [toks[0]]
    # changes.append("\n" + "\n".join(toks[1].strip().split("\n")[0:-1]))
    # changes = ("-" * n).join(changes)

    # with open("docs_rst/latest_changes.rst", "w") as f:
    #     f.write(changes)

    with cd("api-docs-source"):
        ctx.run("rm maml.*.rst", warn=True)
        ctx.run("sphinx-apidoc --separate -P -d 7 -o . -f ../maml")
        ctx.run("rm maml*.tests.*rst", warn=True)
        # for f in glob.glob("maml*.rst"):
        #     newoutput = []
        #     suboutput = []
        #     subpackage = False
        #     with open(f, 'r') as fid:
        #         for line in fid:
        #             clean = line.strip()
        #             if clean == "Subpackages":
        #                 subpackage = True
        #             if not subpackage and not clean.endswith("tests"):
        #                 newoutput.append(line)
        #             else:
        #                 if not clean.endswith("tests"):
        #                     suboutput.append(line)
        #                 if clean.startswith("maml") and not clean.endswith("tests"):
        #                     newoutput.extend(suboutput)
        #                     subpackage = False
        #                     suboutput = []

            # with open(f, 'w') as fid:
            #     fid.write("".join(newoutput))
    ctx.run("sphinx-build -b html api-docs-source api-docs")

    # ctx.run("cp _static/* ../docs/html/_static", warn=True)

    with cd("api-docs"):
        ctx.run("rm -r doctrees", warn=True)

        # This makes sure maml.org works to redirect to the Github page
        # ctx.run("echo \"maml.org\" > CNAME")
        # Avoid the use of jekyll so that _dir works as intended.
        ctx.run("touch .nojekyll")

@task
def update_doc(ctx):
    """
    Update the web documentation.

    :param ctx:
    """
    ctx.run("cp README.rst docs_rst/conf-normal.py docs_rst/conf.py")
    ctx.run("cp docs_rst/conf-normal.py docs_rst/conf.py")
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
def merge_stable(ctx):
    """
    Tag and merge into stable branch.

    :param ctx:
    """
    ctx.run("git commit -a -m \"v%s release\"" % (NEW_VER, ), warn=True)
    ctx.run("git tag -a v%s -m \"v%s release\"" % (NEW_VER, NEW_VER))
    ctx.run("git push --tags")
    ctx.run("git checkout stable")
    ctx.run("git pull")
    ctx.run("git merge master")
    ctx.run("git push")
    ctx.run("git checkout master")


@task
def release_github(ctx):
    """
    Release to Github using Github API.

    :param ctx:
    """
    with open("CHANGES.rst") as f:
        contents = f.read()
    toks = re.split(r"\-+", contents)
    desc = toks[1].strip()
    toks = desc.split("\n")
    desc = "\n".join(toks[:-1]).strip()
    payload = {
        "tag_name": "v" + NEW_VER,
        "target_commitish": "master",
        "name": "v" + NEW_VER,
        "body": desc,
        "draft": False,
        "prerelease": False
    }
    response = requests.post(
        "https://api.github.com/repos/materialsproject/maml/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]})
    print(response.text)


@task
def post_discourse(ctx):
    """
    Post release announcement to http://discuss.matsci.org/c/maml.

    :param ctx:
    """
    with open("CHANGES.rst") as f:
        contents = f.read()
    toks = re.split(r"\-+", contents)
    desc = toks[1].strip()
    toks = desc.split("\n")
    desc = "\n".join(toks[:-1]).strip()
    raw = "v" + NEW_VER + "\n\n" + desc
    payload = {
        "topic_id": 36,
        "raw": raw,
    }
    response = requests.post(
        "https://discuss.matsci.org/c/maml/posts.json",
        data=payload,
        params={
            "api_username": os.environ["DISCOURSE_API_USERNAME"],
            "api_key": os.environ["DISCOURSE_API_KEY"]}
    )
    print(response.text)


@task
def update_changelog(ctx):
    """
    Create a preliminary change log using the git logs.

    :param ctx:
    """
    output = subprocess.check_output(["git", "log", "--pretty=format:%s",
                                      "v%s..HEAD" % CURRENT_VER])
    lines = ["* " + l for l in output.decode("utf-8").strip().split("\n")]
    with open("CHANGES.rst") as f:
        contents = f.read()
    l = "=========="
    toks = contents.split(l)
    head = "\n\nv%s\n" % NEW_VER + "-" * (len(NEW_VER) + 1) + "\n"
    toks.insert(-1, head + "\n".join(lines))
    with open("CHANGES.rst", "w") as f:
        f.write(toks[0] + l + "".join(toks[1:]))
    ctx.run("open CHANGES.rst")


@task
def release(ctx, notest=False, nodoc=False):
    """
    Run full sequence for releasing maml.

    :param ctx:
    :param notest: Whether to skip tests.
    :param nodoc: Whether to skip doc generation.
    """
    ctx.run("rm -r dist build maml.egg-info", warn=True)
    set_ver(ctx)
    if not notest:
        ctx.run("pytest maml")
    publish(ctx)
    if not nodoc:
        # update_doc(ctx)
        make_doc(ctx)
        ctx.run("git add .")
        ctx.run("git commit -a -m \"Update docs\"")
        ctx.run("git push")
    merge_stable(ctx)
    release_github(ctx)
    post_discourse(ctx)


@task
def open_doc(ctx):
    """
    Open local documentation in web browser.

    :param ctx:
    """
    pth = os.path.abspath("docs/_build/html/index.html")
    webbrowser.open("file://" + pth)
