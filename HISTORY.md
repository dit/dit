History
=======
Although the first commit to `dit` was in January of 2013, a much longer
history has influenced the design of `dit`. Here we provide that historical
context.

Originally, `dit` began as a single module in `cmpy`, a
Python package for computational mechanics developed as part of the
Complexity Sciences Center at UC Davis.  In a sense, this was version -2.0.
The commit which introduced `information.py` (on 2007-04-18) included the
following statement:


    Introducing some functions that compute information-theoretic quantities.
    In the future, we might need to formalize a 'distribution' into a
    dictionary-like class.

It was intentionally simple and cludgy, but it did what it needed to
do---providing functions for calculating entropy, mutual information,
conditional entropy, etc.  As `cmpy` grew, this single module eventually
became a subpackage which included even more functionality.  However, it was
already abundantly clear that a more formal information theory package
would be needed.

This brings us to a fairly innocent commit message in January of 2010:

    first stab at a Distribution class

This was a complete rewrite and, in hindsight, could be seen as `dit -1.0`.
It was fresh, formal and exactly what `cmpy` needed.  Its development
continued and was eventually split off into a new `infotheory` subpackage. With
the subpackage in place, features blossomed over the next few years to include
multivariate generalizations of Shannon's information theory as well algorithms
for calculating candidates measures for redundancy, synergy, and unique
information.

Still, the growth of the `infotheory` subpackage was quite organic, and many
of the original designs has been diluted due to unexpected use cases.
Discussions on a rewrite were commonplace, but design is always tricky and so,
a rewrite was delayed for some time.

In September of 2012, the first bits of `dit` were discussed over GChat
(now known as Google Hangouts), and written to file.  Then finally, the
first commit made its way into the record books in January of 2013, now
available as a completely separate package.  Its development was slow but
steady, and picked up quite a bit around September 2013.

This short outline is no where near detailed enough to give justice to the
work and decisions made along the way, but `dit` has benefited immensely from
these prior experiences---they have helped determine the API, typical use
patterns, and much more.  Given the history, its not unlikely that `dit` might
eventually be rewritten as well.  Until then, it will grow organically
to meet the needs of those who use it.

Support
=======
`dit` and those who have contributed to `dit` have received support throughout
the years from a variety of sources.  We list them below.  If you have
contributed to `dit` and a support acknowledgment does not appear below, please
help us remedy the situation, and similarly, please let us know if you'd like
something modified or corrected.


Research Groups
---------------

    Complexity Sciences Center
    Department of Physics
    University of California-Davis
    PI: James P. Crutchfield

    Center for Complexity and Collective Computation
    Wisconsin Institute for Discovery
    University of Wisconsin-Madison
    PI: David C. Krakauer and Jessica C. Flack

    Department of Computer Science
    University of Colorado-Boulder
    PI: Elizabeth Bradley


Funding
-------

    ARO grant W911NF-12-1-0288
    DARPA Physical Intelligence Subcontract No. 9060-000709
    NSF Grant No. PHY-0748828
    John Templeton Foundation

