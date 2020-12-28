[Relations]
posts_posttypeid(PostId, nominalTarget)
badges_userid(BadgeId, UserId)
badges_name(BadgeId, nominal1)
badges_date(BadgeId, numeric2)
comments_postid(CommentId, PostId)
comments_score(CommentId, numeric3)
comments_creationdate(CommentId, numeric4)
comments_userdisplayname(CommentId, nominal5)
comments_userid(CommentId, UserId)
posthistory_posthistorytypeid(UpdateId, PosthistoryTypeId)
posthistory_postid(UpdateId, PostId)
posthistory_creationdate(UpdateId, numeric6)
posthistory_userid(UpdateId, UserId)
posthistory_userdisplayname(UpdateId, nominal7)
posthistory_comment(UpdateId, nominal8)
posts_acceptedanswerid(PostId, AcceptedAnswerId)
posts_parentid(PostId, ParentId)
posts_creationdate(PostId, numeric9)
posts_score(PostId, numeric10)
posts_viewcount(PostId, numeric11)
posts_owneruserid(PostId, UserId)
posts_ownerdisplayname(PostId, nominal12)
posts_lasteditoruserid(PostId, Userid)
posts_lasteditordisplayname(PostId, nominal13)
posts_lasteditdate(PostId, numeric15)
posts_lastactivitydate(PostId, numeric17)
posts_title(PostId, nominal19)
posts_tags(PostId, nominal21)
posts_answercount(PostId, numeric23)
posts_commentcount(PostId, nominal25)
posts_favoritecount(PostId, numeric27)
posts_closeddate(PostId, numeric29)
posts_communityowneddate(PostId, numeric31)
tags_tagname(TagId, nominal32)
tags_count(TagId, numeric33)
tags_excerptpostid(TagId, PostId)
tags_wikipostid(TagId, WikiPostId)
users_creationdate(UserId, numeric0)
users_displayname(UserId, nominal34)
users_lastaccessdate(UserId, numeric35)
users_location(UserId, nominal36)
users_views(UserId, numeric37)
users_upvotes(UserId, numeric138)
users_downvotes(UserId, numeric39)
users_emailhash(UserId, nominal40)
users_age(UserId, numeric41)
users_accountid(UserId, AccountId)
users_reputation(UserId, nominal42)
votes_postid(VoteId, PostId)
votes_votetypeid(VoteId, nominal43)
votes_userid(VoteId, UserId)
votes_creationdate(VoteId, numeric44)
votes_bountyamount(VoteId, numeric45)
[Aggregates]
sum
count
[AtomTests]
badges_userid(new, old)
badges_name(old, new)
//badges_date(old, new)
comments_postid(new, old)
comments_score(old, new)
//comments_creationdate(old, new)
comments_userdisplayname(old, new)
comments_userid(old, new)
comments_userid(new, old)
posthistory_posthistorytypeid(old, new)
posthistory_postid(new, old)
//posthistory_creationdate(old, new)
posthistory_userid(old, new)
//posthistory_userdisplayname(old, new)
//posthistory_comment(old, new)
posts_acceptedanswerid(old, new)
posts_parentid(old, new)
//posts_creationdate(old, new)
posts_score(old, new)
posts_viewcount(old, new)
posts_owneruserid(old, new)
posts_ownerdisplayname(old, new)
posts_lasteditoruserid(old, new)
posts_lasteditordisplayname(old, new)
//posts_lasteditdate(old, new)
//posts_lastactivitydate(old, new)
posts_title(old, new)
posts_tags(old, new)
posts_answercount(old, new)
posts_commentcount(old, new)
posts_favoritecount(old, new)
//posts_closeddate(old, new)
//posts_communityowneddate(old, new)
tags_tagname(old, new)
tags_count(old, new)
tags_excerptpostid(new, old)
tags_wikipostid(old, new)
//users_creationdate(old, new)
users_displayname(old, new)
//users_lastaccessdate(old, new)
users_location(old, new)
users_views(old, new)
users_upvotes(old, new)
users_downvotes(old, new)
users_emailhash(old, new)
users_age(old, new)
users_accountid(old, new)
users_reputation(old, new)
votes_postid(new, old)
votes_votetypeid(old, new)
votes_userid(old, new)
//votes_creationdate(old, new)
votes_bountyamount(old, new)