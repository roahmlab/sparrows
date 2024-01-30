# Getting Started

This project is designed to be both configurable and lightweight.
Most likely if you're using this as a project specific page, you only need one markdown file to describe the whole page.

**Do NOT fork this repo to use the template!**
This repo holds the remote-theme for GitHub Pages in an orphan branch and as a result, forking it means you are actually downloading much more than you need (even if the theme isn't that big).
Besides, it makes it more difficult to get running as a project page anyway!
All you should do is download the archive for one of the example branches (where you're reading this from right now!) and upload these files.
These examples are close to the minimal files needed.

If you are planning to use this to make an organization specific template, then forking is reasonable.
Just make sure to change the `remote_theme` key inside of `_config.yml` to match your organization's setup.
More information on that can be found at the [jekyll-remote-theme repo](https://github.com/benbalter/jekyll-remote-theme).

## Create an orphan branch and ensure that it is clean

You'll want to keep your website code separate from the history of your actual project.
To do this, first create an orphan branch and ensure it's clean of any other files.
It may be worth recloning your target repository onto another system to ensure you aren't adding extra gitignored files.

```bash
git switch --orphan gh-pages
# Check if you have any extra files
git status
# And make sure to delete them
```

Alternately, you can create a new folder and initialize it as a new git repo.
In this scenario, you can set the same remote repository as the upstream repo, and just make sure to commit the website files to a dedicated branch separate from the others.

## Extract these files to your desired location

From within your repository, execute the following commands.

```bash
wget "https://github.com/BuildingAtom/project-pages/archive/refs/heads/main-v1-examples.zip" -O /tmp/project-pages-starter.zip
unzip /tmp/project-pages-starter.zip -d /tmp/
rsync -a /tmp/project-pages-main-v1-examples/ ./
# Verify that you have .gitignore and .github copied
ls -la
```

I suggest `rsync` here because it's able to copy the hidden files that globbing with `mv` or `cp` run the risk of ignoring.
Windows users shouldn't have a problem with using Windows Explorer as that doesn't hide `.` prefixed files.
Just download the zip from the top if you're a Windows user and extract it to your desired repository location.

## Delete extra examples

These examples are obviously going to remain accessible through this repo.
Go ahead and delete them before you commit anything.

```bash
rm armour-dev.markdown sel_map.markdown
```

At this point, it's reasonable to make an initial commit on the orphan branch, so go ahead and do that.
A workflow is included in these files already, so this should trigger a github action, but it will likely fail to deploy.
Verify that your Pages section in Settings is set correctly as [per this tutorial](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow).
*Note that you shouldn't need to create a new workflow.*
After that, verify that you have enabled deploy permissions for the `gh-pages` branch, or whichever name you used, to the `github-pages` environment under the Environments tab in Settings.

```bash
git commit -a -m "Initial project-pages template commit."
git push -u origin gh-pages
```

## (Optional) Get Jekyll running locally

Follow the [Jekyll installation instructions](https://jekyllrb.com/docs/installation/).
Then, inside of the repo folder, install and setup the required Gems with `Bundler`.
After that, you can launch the server (preferably with livereload because that makes life easier).

```bash
bundle install
bundle exec jekyll server --livereload
```

This way, any edits you make to anything but the Gemfiles and `_config.yml` will reload your browser window for you.

---

# Configuration

**There are 4 files:**
* `_config.yml`: Holds core key values that you'll want to configure first.
These are not reloaded during live preview and require a restart of the server whenever they're changed.
* `index.markdown`: Although this isn't solely a configuration file, the first part (the Front Matter) holds a lot of additional metadata that will need filling out.
There should be ample comments for both of the above.
* `_data/fonts.yml`: This is a font configuration file based on Google Fonts.
From the Google Fonts page, after you have selected whatever fonts you want, go to your selections in the top right, then go to the `Use on the web` section.
In that box there should be 3 lines.
Just copy the string associated with the last `href` tag and add that to the `google-fonts-href` list in this file.
The rest of the file is fairly self-explainable.
* `css/stylesheet.scss`: This holds variable definitions for a lot of the parameters that are used throughout the scss files.
Each of these should be self-explanatory, but change these to change the appearance of the site.

**Note**: In addition to `css/stylesheet.scss`, you can make your own `css/overrides.css` or `css/overrides.scss` file.
If it exists, this file is imported after `css/stylesheeet.scss` and can be used to override some of the style I have written.

If you wish to add plugins, check [GitHub Pages Dependency Versions](https://pages.github.com/versions/) to make sure your desired plugin is supported!

---

At this point, you have all the most basic initial configuration down!
Read through `index.markdown` to see what can be done in terms of actual contet.
The kramdown interpreter is used here, with the rouge code-highlighter.
