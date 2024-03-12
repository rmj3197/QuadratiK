# Contributing to QuadratiK

First of all, thanks for considering contributing to QuadratiK!

[repo]: https://github.com/rmj3197/QuadratiK
[issues]: https://github.com/rmj3197/QuadratiK/issues
[new_issue]: https://github.com/rmj3197/QuadratiK/issues/new
[website]: https://quadratik.readthedocs.io/en/latest/
[citation]: https://arxiv.org/abs/2402.02290
[conduct]: https://quadratik.readthedocs.io/en/latest/development/CONTRIBUTING.html
[bug_report]: https://github.com/rmj3197/QuadratiK/issues/new?assignees=&labels=Bug%2CNeeds+Triage&projects=&template=bug_report.yml
[doc_improvement]: https://github.com/rmj3197/QuadratiK/issues/new?assignees=&labels=Documentation%2CNeeds+Triage&projects=&template=documentation_improvement.yml
[email]: mailto:raktimmu@buffalo.edu

## Code of conduct

Please note that this project is released with a [Contributor Code of Conduct][conduct]. By participating in this project you agree to abide by its terms.

## How you can contribute

There are several ways you can contribute to this project.

### Share the love ❤️

Think QuadratiK is useful? Let others discover it, by telling them in person, via Twitter, ResearchGate or a blog post.

Using QuadratiK for a paper you are writing? Consider [citing it][citation].

### Ask a question ⁉️

Using QuadratiK and got stuck? Browse the [documentation][website] to see if you can find a solution. Still stuck? Post your question as an [issue on GitHub][new_issue]. While we cannot offer user support, we'll try to do our best to address it, as questions often lead to better documentation or the discovery of bugs.

Want to ask a question in private? Contact the package maintainer by [mail][email].

### Propose an idea 💡

Have an idea for a new QuadratiK feature? Take a look at the [documentation][website] and [issue list][issues] to see if it isn't included or suggested yet. If not, suggest your idea as an [issue on GitHub][new_issue]. While we can't promise to implement your idea, it helps to:

- Explain in detail how it would work.
- Keep the scope as narrow as possible.

See below if you want to contribute code for your idea as well.

### Report a bug 🐛

Using QuadratiK and discovered a bug? That's annoying! Don't let others have the same experience and report it as an [issue on GitHub][new_issue] so we can fix it. A good bug report makes it easier for us to do so, please try to give as much detail as possible, at [Bug Report][bug_report].

### Improve the documentation 📖

Noticed a typo on the website? Think a function could use a better example? Good documentation makes all the difference, so your help to improve it is very welcome! Submit a issue here [Documentation Improvement][doc_improvement].

#### API documentation

The API documentation is built automatically from the docstrings of classes, functions, etc. in the source files. The docs are built with sphinx and use the "sphinx_book_theme" theme. If you have the dependencies installed (tool.poetry.group.doc.dependencies inside pyproject.toml) you can build the documentation locally with `make html` in the /doc directory. Opening the /doc/build/index.html file with a browser will then allow you to browse the documentation and check your contributions locally.

- Go to `QuadratiK/` directory in the [code repository][repo].
- Look for the file with the name of the function.
- [Propose a file change](https://help.github.com/articles/editing-files-in-another-user-s-repository/) to update the function documentation in the roxygen comments (starting with `#'`).

### Contribute code 📝

Care to fix bugs or implement new functionality for QuadratiK? Awesome! 👏 Have a look at the [issue list][issues] and leave a comment on the things you want to work on. See also the development guidelines below.

## Development guidelines

We try to follow the [GitHub flow](https://guides.github.com/introduction/flow/) for development.

1. Fork [this repo][repo] and clone it to your computer. To learn more about this process, see [this guide](https://guides.github.com/activities/forking/).
2. If you have forked and cloned the project before and it has been a while since you worked on it, [pull changes from the original repo](https://help.github.com/articles/merging-an-upstream-repository-into-your-fork/) to your clone by using `git pull upstream master`.
3. Open the folder on your local machine using any code editor.
4. Make your changes:

   - Write your code.
   - Test your code (bonus points for adding unit tests).
   - Document your code (see function documentation above).
   - Check your code with `pytest`.

5. Commit and push your changes.
6. Submit a [pull request](https://guides.github.com/activities/forking/#making-a-pull-request).

## Future Developments

QuadratiK was first released on PyPI on February 03, 2024. Regular updates and bug fixes are planned to continually enhance the package's functionality and user experience. We are actively planning to include additional methods based on the kernel-based quadratic-distance. One of our primary goals is to make QuadratiK increasingly user-friendly, with improvements to the user experience and the layout of the outputs. User feedback is highly valued and will be a key driver of future development. This Life Cycle Statement is subject to periodic review and will be updated to reflect the evolving nature of QuadratiK. 